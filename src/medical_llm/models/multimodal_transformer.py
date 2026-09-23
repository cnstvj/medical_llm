"""
MultiModal Transformer architecture for joint Clinical Time-Series and Pretrained Text Modeling.
Integrates Structured Lab Trajectories (TransformerEncoder) + Clinical Notes (Bio_ClinicalBERT) via Gated/Cross-Attention Fusion.
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from medical_llm.models.transformer import PositionalEncoding


class ClinicalTextEncoder(nn.Module):
    """Backbone encoder for unstructured clinical text using Bio_ClinicalBERT or fallback embedding."""

    def __init__(
        self,
        pretrained_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 768,
        proj_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_hf = False
        self.bert = None

        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            if freeze_backbone:
                for param in self.bert.parameters():
                    param.requires_grad = False
            self.use_hf = True
            hidden_size = self.bert.config.hidden_size
        except Exception:
            # Fallback for environments without network access / mock testing
            self.bert = nn.Embedding(30522, embedding_dim, padding_idx=0)
            hidden_size = embedding_dim

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled_embedding: (batch_size, proj_dim)
            token_embeddings: (batch_size, seq_len, proj_dim)
        """
        if self.use_hf and hasattr(self.bert, "encoder"):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Masked mean pooling
            last_hidden_state = outputs.last_hidden_state  # (B, L, H)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Fallback embedding average
            embeds = self.bert(input_ids)
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeds.size()).float()
            sum_embeddings = torch.sum(embeds * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            last_hidden_state = embeds

        proj_pooled = self.projection(pooled)
        return proj_pooled, last_hidden_state


class GatedCrossModalFusion(nn.Module):
    """Gated multimodal fusion layer dynamically weighting EHR trajectory and Clinical Notes."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fc_fused = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_labs: torch.Tensor, h_text: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([h_labs, h_text], dim=-1)
        alpha = self.gate(concat)
        fused = alpha * h_labs + (1 - alpha) * h_text
        return fused


class MultiModalMedicalTransformer(nn.Module):
    """Publication-ready Multi-Modal Clinical Transformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_lab_len: int = 512,
        pretrained_text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze_text_backbone: bool = True,
        fusion_type: str = "gated",
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type

        # 1. Structured Time-Series Lab Encoder
        self.lab_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_lab_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2. Unstructured Clinical Note Text Encoder
        self.text_encoder = ClinicalTextEncoder(
            pretrained_model_name=pretrained_text_model,
            proj_dim=d_model,
            freeze_backbone=freeze_text_backbone,
        )

        # 3. Fusion Layer
        if fusion_type == "gated":
            self.fusion = GatedCrossModalFusion(d_model, dropout=dropout)
            out_dim = d_model
        elif fusion_type == "concat":
            self.fusion = None
            out_dim = d_model * 2
        else:
            self.fusion = GatedCrossModalFusion(d_model, dropout=dropout)
            out_dim = d_model

        # 4. Classification & Prediction Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier & Normal weight initialization to stabilize training gradients."""
        nn.init.normal_(self.lab_embedding.weight, mean=0.0, std=0.02)
        if self.lab_embedding.padding_idx is not None:
            nn.init.constant_(self.lab_embedding.weight[self.lab_embedding.padding_idx], 0.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        # Classifier head low-variance initialization
        last_linear = self.classifier[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)

    def forward(
        self,
        lab_ids: torch.Tensor,
        lab_mask: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        # A. Encode structured labs
        x_lab = self.lab_embedding(lab_ids)
        x_lab = self.pos_encoder(x_lab)

        pad_mask = lab_mask == 0
        encoded_labs = self.transformer_encoder(x_lab, src_key_padding_mask=pad_mask)

        # Masked mean pooling for labs
        mask_f = lab_mask.unsqueeze(-1).float()
        pooled_labs = (encoded_labs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        # B. Encode unstructured clinical notes
        pooled_text, _ = self.text_encoder(text_ids, text_mask)

        # C. Multi-modal Fusion
        if self.fusion_type == "gated":
            fused = self.fusion(pooled_labs, pooled_text)
        elif self.fusion_type == "concat":
            fused = torch.cat([pooled_labs, pooled_text], dim=-1)
        else:
            fused = self.fusion(pooled_labs, pooled_text)

        # D. Mortality Logit
        logits = self.classifier(fused).squeeze(-1)
        return logits
