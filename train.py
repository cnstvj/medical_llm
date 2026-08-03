# This is a new branch

import math
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import subprocess

from dataset import get_dataloaders
from model import MedicalTransformer
from sklearn.metrics import roc_auc_score, f1_score

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DUMMY_MODE = "--dummy" in sys.argv


def prepare_data():
    """Run preprocessing and sequence building if needed."""

    env = os.environ.copy()
    if DUMMY_MODE:
        print("[DUMMY MODE] Generating dummy CSV files...")
        subprocess.run(
            [sys.executable, "generate_dummy_data.py"],
            check=True
        )
        env["DUMMY_MODE"] = "1"

    if not os.path.exists("sequence_dataset.json"):
        print("Running preprocessing...")
        subprocess.run(
            [sys.executable, "preprocessing.py"],
            env=env,
            check=True
        )
        print("Running sequence builder...")
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
    print(f"\n{'='*50}")
    print(f"TRAINING ON {mode_label} DATASET")
    print(f"{'='*50}\n")

    prepare_data()

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        "sequence_dataset.json"
    )

    model = MedicalTransformer(len(vocab)).to(device)

    labels = train_loader.dataset.labels
    pos = sum(labels)
    neg = len(labels) - pos

    pos_weight = torch.tensor([neg/max(pos,1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

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

        print(f"Epoch {epoch+1}: AUROC={val_auc:.4f} F1={val_f1:.4f}")

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
    print(f"\nTEST AUROC={test_auc:.4f} F1={test_f1:.4f}")

if __name__ == "__main__":
    train()
