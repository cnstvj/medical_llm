"""
Unsupervised / Self-Supervised Masked Event Modeling (MEM) Module.
Pretrains clinical trajectory transformer encoders on unlabeled lab events
via masked token reconstruction (BERT-style self-supervision).
"""

import random
from typing import Dict, Any, Tuple, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from medical_llm.models.transformer import PositionalEncoding
from medical_llm.models.time2vec import Time2Vec


class MaskedTrajectoryDataset(Dataset):
    """
    Unsupervised dataset that dynamically masks tokens for self-supervised pretraining.
    Supports dynamic masking probabilities (15% to 20%) over lab trajectory tokens:
    - Dynamic percentage of tokens selected for prediction
    - Of selected: 80% replaced with [MASK], 10% random token, 10% kept unchanged
    """

    def __init__(
        self,
        sequences: List[List[str]],
        vocab: Dict[str, int],
        mask_prob: Union[float, Tuple[float, float]] = (0.15, 0.20),
        max_len: int = 512,
    ):
        self.sequences = sequences
        self.vocab = vocab
        self.inv_vocab = {idx: tok for tok, idx in vocab.items()}
        self.mask_prob = mask_prob
        self.max_len = max_len

        self.pad_id = vocab.get("[PAD]", 0)
        self.unk_id = vocab.get("[UNK]", 1)
        self.cls_id = vocab.get("[CLS]", 2)
        self.sep_id = vocab.get("[SEP]", 3)
        self.mask_id = vocab.get("[MASK]", 4 if len(vocab) > 4 else 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        seq_ids = [self.vocab.get(tok, self.unk_id) for tok in seq]

        if len(seq_ids) > self.max_len - 1:
            seq_ids = seq_ids[:(self.max_len - 1)]

        seq_ids = [self.cls_id] + seq_ids

        input_ids = list(seq_ids)
        targets = [-100] * len(seq_ids)  # PyTorch ignore_index

        # Dynamic masking probability (sample per trajectory sequence)
        if isinstance(self.mask_prob, (tuple, list)):
            effective_mask_prob = random.uniform(self.mask_prob[0], self.mask_prob[1])
        else:
            effective_mask_prob = float(self.mask_prob)

        # Masking tokens (excluding special tokens)
        masked_indices = []
        for i in range(1, len(seq_ids)):
            token_id = seq_ids[i]
            if token_id in (self.cls_id, self.sep_id, self.pad_id):
                continue
            if random.random() < effective_mask_prob:
                masked_indices.append(i)

        # Ensure at least 1 token is masked if sequence has non-special tokens
        if not masked_indices and len(seq_ids) > 1:
            masked_indices.append(random.randint(1, len(seq_ids) - 1))

        for i in masked_indices:
            targets[i] = seq_ids[i]
            prob = random.random()
            if prob < 0.8:
                input_ids[i] = self.mask_id
            elif prob < 0.9:
                input_ids[i] = random.randint(1, len(self.vocab) - 1)
            # Remaining 10% keep token unchanged

        pad_len = self.max_len - len(seq_ids)
        attention_mask = [1] * len(seq_ids) + [0] * pad_len
        input_ids += [self.pad_id] * pad_len
        targets += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


class MaskedTrajectoryPretrainer(nn.Module):
    """
    Self-supervised sequence encoder head for reconstructing masked clinical lab events.
    Enables pretraining without any binary mortality labels.
    """

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
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Unsupervised prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)

        pad_mask = attention_mask == 0
        hidden = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        logits = self.head(hidden)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
