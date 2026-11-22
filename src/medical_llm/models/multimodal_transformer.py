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
from medical_llm.models.time2vec import Time2Vec


class ClinicalTextEncoder(nn.Module):
    """Backbone encoder for unstructured clinical text using Bio_ClinicalBERT with optional LoRA adapter or fallback."""

    def __init__(
        self,
        pretrained_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embedding_dim: int = 768,
        proj_dim: int = 128,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_hf = False
        self.bert = None

        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            
            if use_lora:
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    peft_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_r * 2,
                        target_modules=["query", "value"],
                        lora_dropout=0.1,
                        bias="none",
                    )
                    self.bert = get_peft_model(self.bert, peft_config)
                except Exception:
                    # Fallback to freezing if LoRA config fails
                    if freeze_backbone:
                        for param in self.bert.parameters():
                            param.requires_grad = False
            elif freeze_backbone:
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
            proj_tokens: (batch_size, seq_len, proj_dim)
        """
        if self.use_hf and hasattr(self.bert, "config"):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # (B, L, H)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            embeds = self.bert(input_ids)
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeds.size()).float()
            sum_embeddings = torch.sum(embeds * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            last_hidden_state = embeds

        proj_pooled = self.projection(pooled)
        proj_tokens = self.projection(last_hidden_state)
        return proj_pooled, proj_tokens


class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention between EHR Trajectories and Clinical Notes.
    Enables lab sequence tokens to attend to clinical notes and notes to attend to lab events.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_labs = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm_labs = nn.LayerNorm(d_model)
        self.norm_text = nn.LayerNorm(d_model)

        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_labs: torch.Tensor,
        h_text: torch.Tensor,
        lab_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_labs: (batch_size, lab_seq_len, d_model)
            h_text: (batch_size, text_seq_len, d_model)
        Returns:
            fused_representation: (batch_size, d_model)
        """
        lab_pad_mask = None
        if lab_mask is not None:
            # MultiheadAttention requires not all tokens to be masked in a row
            l_mask_bool = (lab_mask == 0)
            if not l_mask_bool.all(dim=-1).any():
                lab_pad_mask = l_mask_bool

        text_pad_mask = None
        if text_mask is not None:
            t_mask_bool = (text_mask == 0)
            if not t_mask_bool.all(dim=-1).any():
                text_pad_mask = t_mask_bool

        # Labs query text
        attn_labs, _ = self.cross_attn_labs(
            query=h_labs, key=h_text, value=h_text, key_padding_mask=text_pad_mask
        )
        h_labs_fused = self.norm_labs(h_labs + attn_labs)

        # Text queries labs
        attn_text, _ = self.cross_attn_text(
            query=h_text, key=h_labs, value=h_labs, key_padding_mask=lab_pad_mask
        )
        h_text_fused = self.norm_text(h_text + attn_text)

        # Masked pooling
        if lab_mask is not None:
            m_l = lab_mask.unsqueeze(-1).float()
            pooled_l = (h_labs_fused * m_l).sum(dim=1) / m_l.sum(dim=1).clamp(min=1)
        else:
            pooled_l = h_labs_fused.mean(dim=1)

        if text_mask is not None:
            m_t = text_mask.unsqueeze(-1).float()
            pooled_t = (h_text_fused * m_t).sum(dim=1) / m_t.sum(dim=1).clamp(min=1)
        else:
            pooled_t = h_text_fused.mean(dim=1)

        fused = self.fusion_fc(torch.cat([pooled_l, pooled_t], dim=-1))
        return fused


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
    """Publication-ready Multi-Modal Clinical Transformer with continuous-time embedding, LoRA, and Cross-Attention."""

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
        use_lora: bool = False,
        lora_r: int = 8,
        fusion_type: str = "cross_attention",
        use_time2vec: bool = True,
        time_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.use_time2vec = use_time2vec

        # 1. Structured Time-Series Lab Encoder with Time2Vec
        self.lab_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_lab_len)

        if use_time2vec:
            self.time2vec = Time2Vec(time_dim=time_dim)
            self.time_proj = nn.Linear(time_dim, d_model)
        else:
            self.time2vec = None
            self.time_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 2. Unstructured Clinical Note Text Encoder
        self.text_encoder = ClinicalTextEncoder(
            pretrained_model_name=pretrained_text_model,
            proj_dim=d_model,
            freeze_backbone=freeze_text_backbone,
            use_lora=use_lora,
            lora_r=lora_r,
        )

        # 3. Fusion Layer
        if fusion_type == "cross_attention":
            self.fusion = BidirectionalCrossAttentionFusion(d_model, nhead=nhead, dropout=dropout)
            out_dim = d_model
        elif fusion_type == "gated":
            self.fusion = GatedCrossModalFusion(d_model, dropout=dropout)
            out_dim = d_model
        elif fusion_type == "concat":
            self.fusion = None
            out_dim = d_model * 2
        else:
            self.fusion = BidirectionalCrossAttentionFusion(d_model, nhead=nhead, dropout=dropout)
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
        delta_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # A. Encode structured labs
        x_lab = self.lab_embedding(lab_ids)
        x_lab = self.pos_encoder(x_lab)

        if self.use_time2vec and delta_times is not None:
            t_emb = self.time_proj(self.time2vec(delta_times))
            x_lab = x_lab + t_emb

        pad_mask = lab_mask == 0
        encoded_labs = self.transformer_encoder(x_lab, src_key_padding_mask=pad_mask)

        # B. Encode unstructured clinical notes
        pooled_text, proj_text_tokens = self.text_encoder(text_ids, text_mask)

        # C. Multi-modal Fusion
        if self.fusion_type == "cross_attention":
            fused = self.fusion(encoded_labs, proj_text_tokens, lab_mask=lab_mask, text_mask=text_mask)
        elif self.fusion_type == "gated":
            mask_f = lab_mask.unsqueeze(-1).float()
            pooled_labs = (encoded_labs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            fused = self.fusion(pooled_labs, pooled_text)
        elif self.fusion_type == "concat":
            mask_f = lab_mask.unsqueeze(-1).float()
            pooled_labs = (encoded_labs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            fused = torch.cat([pooled_labs, pooled_text], dim=-1)
        else:
            fused = self.fusion(encoded_labs, proj_text_tokens, lab_mask=lab_mask, text_mask=text_mask)

        # D. Mortality Logit
        logits = self.classifier(fused).squeeze(-1)
        return logits
