"""
Transformer architecture for medical sequence modeling and mortality classification.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module with dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MedicalTransformer(nn.Module):
    """Stabilized Pre-LN Transformer Encoder model for medical sequence mortality classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

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

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier & Normal weight initialization to stabilize training gradients."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        # Output head low-variance initialization
        last_linear = self.head[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Scale embedding by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        pad_mask = mask == 0

        encoded = self.transformer_encoder(x, src_key_padding_mask=pad_mask)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        logits = self.head(pooled).squeeze(-1)
        return logits
