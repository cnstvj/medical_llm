"""
Trainer module for training the MedicalTransformer model.
"""

import math
import os
import sys
import subprocess
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score

from medical_llm.config import DEFAULT_CONFIG, PipelineConfig
from medical_llm.utils.logging import log_event
from medical_llm.data.dataset import get_dataloaders
from medical_llm.models.transformer import MedicalTransformer


def prepare_data(dummy_mode: bool = False, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Runs dummy data generation and preprocessing/sequence builder pipelines if sequence dataset is missing."""
    env = os.environ.copy()
    if dummy_mode:
        log_event("[DUMMY MODE] Generating dummy CSV files...")
        subprocess.run([sys.executable, "generate_dummy_data.py"], check=True)
        env["DUMMY_MODE"] = "1"

    if not os.path.exists(config.sequence_dataset_file):
        log_event("Running preprocessing...")
        subprocess.run([sys.executable, "preprocessing.py"], env=env, check=True)
        log_event("Running sequence builder...")
        subprocess.run([sys.executable, "sequence_builder.py"], check=True)


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluates model performance and returns AUROC and F1 metrics."""
    model.eval()

    probs = []
    labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    preds = [1 if p > 0.5 else 0 for p in probs]

    try:
        auc = roc_auc_score(labels, probs)
        if math.isnan(auc):
            auc = 0.0
    except Exception:
        auc = 0.0

    return auc, f1_score(labels, preds, zero_division=0)


def train(dummy_mode: bool = False, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Main training loop for MedicalTransformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_label = "DUMMY" if dummy_mode else "REAL"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_event("\n" + "=" * 50)
    log_event(f"[{timestamp}] EVENT: TRAIN PIPELINE RUN ({mode_label} DATASET)")
    log_event("=" * 50 + "\n")

    prepare_data(dummy_mode=dummy_mode, config=config)

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        config.sequence_dataset_file, batch_size=config.batch_size, max_len=config.max_len
    )

    model = MedicalTransformer(
        len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    labels = train_loader.dataset.labels
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive

    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    log_event(f"Training samples: {len(labels)} (Pos: {num_positive}, Neg: {num_negative})")
    log_event(f"Class imbalance weight (pos_weight): {pos_weight.item():.4f}\n")

    best_auc = -1.0

    for epoch in range(config.epochs):
        model.train()

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        val_auc, val_f1 = evaluate(model, val_loader, device)

        log_event(f"Epoch {epoch+1:2d}/{config.epochs}: AUROC={val_auc:.4f} F1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                },
                "best_model.pth",
            )

    checkpoint = torch.load("best_model.pth", map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    test_auc, test_f1 = evaluate(model, test_loader, device)
    log_event(f"\nTEST RESULT: AUROC={test_auc:.4f} F1={test_f1:.4f}")
    log_event("Model saved to best_model.pth\n")
