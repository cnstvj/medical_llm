# train.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import subprocess
from datetime import datetime

from dataset import get_dataloaders
from model import MedicalTransformer
from sklearn.metrics import roc_auc_score, f1_score

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DUMMY_MODE = "--dummy" in sys.argv
LOG_FILE = "log.txt"


# ------------------------------------------------------------------
# Logging Segment
# ------------------------------------------------------------------
def log_event(message=""):
    """Prints message to stdout and appends to log.txt."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def prepare_data():
    """Run preprocessing and sequence building if needed."""

    env = os.environ.copy()
    if DUMMY_MODE:
        log_event("[DUMMY MODE] Generating dummy CSV files...")
        subprocess.run(
            [sys.executable, "generate_dummy_data.py"],
            check=True
        )
        env["DUMMY_MODE"] = "1"

    if not os.path.exists("sequence_dataset.json"):
        log_event("Running preprocessing...")
        subprocess.run(
            [sys.executable, "preprocessing.py"],
            env=env,
            check=True
        )
        log_event("Running sequence builder...")
        subprocess.run(
            [sys.executable, "sequence_builder.py"],
            check=True
        )


def evaluate(model, loader, device):
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
    except:
        auc = 0.0

    return auc, f1_score(labels, preds, zero_division=0)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_label = "DUMMY" if DUMMY_MODE else "REAL"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Logging event start
    log_event("\n" + "=" * 50)
    log_event(f"[{timestamp}] EVENT: TRAIN PIPELINE RUN ({mode_label} DATASET)")
    log_event("=" * 50 + "\n")

    prepare_data()

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        "sequence_dataset.json"
    )

    model = MedicalTransformer(len(vocab)).to(device)

    labels = train_loader.dataset.labels
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive

    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    log_event(f"Training samples: {len(labels)} (Pos: {num_positive}, Neg: {num_negative})")
    log_event(f"Class imbalance weight (pos_weight): {pos_weight.item():.4f}\n")

    best_auc = -1

    for epoch in range(10):
        model.train()

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x, mask)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        val_auc, val_f1 = evaluate(model, val_loader, device)

        log_event(f"Epoch {epoch+1:2d}/10: AUROC={val_auc:.4f} F1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
            }, "best_model.pth")

    checkpoint = torch.load("best_model.pth", map_location=device)
    # Support both old (raw state_dict) and new (dict with vocab) format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    test_auc, test_f1 = evaluate(model, test_loader, device)
    log_event(f"\nTEST RESULT: AUROC={test_auc:.4f} F1={test_f1:.4f}")
    log_event("Model saved to best_model.pth\n")


if __name__ == "__main__":
    train()
