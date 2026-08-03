# test.py
# =======
# Evaluate saved fold models on the dataset.
#
# Usage:
#   python test.py            # evaluate on real dataset
#   python test.py --dummy    # evaluate on dummy dataset

import os
import sys
import glob
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from dataset import MedicalDataset, build_vocab, load_dataset_json
from model import MedicalTransformer

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DUMMY_MODE = "--dummy" in sys.argv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------
def prepare_data():
    """Generate dummy CSVs + run preprocessing/sequence builder if needed."""

    env = os.environ.copy()

    if DUMMY_MODE:
        print("[DUMMY MODE] Generating dummy CSV files...")
        subprocess.run(
            [sys.executable, "generate_dummy_data.py"],
            check=True,
        )
        env["DUMMY_MODE"] = "1"

    if not os.path.exists("sequence_dataset.json"):
        print("Running preprocessing...")
        subprocess.run(
            [sys.executable, "preprocessing.py"],
            env=env,
            check=True,
        )
        print("Running sequence builder...")
        subprocess.run(
            [sys.executable, "sequence_builder.py"],
            check=True,
        )


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------
def evaluate_model(model, loader):
    """Run inference and return (probs, labels)."""

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(x, mask)
            p = torch.sigmoid(logits)

            all_probs.extend(p.cpu().numpy())
            all_labels.extend(y.numpy())

    return all_probs, all_labels


def compute_metrics(probs, labels):
    """Compute AUROC, best-threshold F1, precision, recall, confusion matrix."""

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.96, 0.05):
        preds = [1 if p > threshold else 0 for p in probs]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    preds = [1 if p > best_threshold else 0 for p in probs]

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "auroc": auc,
        "f1": best_f1,
        "precision": precision,
        "recall": recall,
        "threshold": best_threshold,
        "confusion_matrix": cm,
    }


def print_metrics(metrics, header=""):
    if header:
        print(f"\n{'='*40}")
        print(header)
        print("=" * 40)

    print(f"AUROC     : {metrics['auroc']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Threshold : {metrics['threshold']:.2f}")
    print(f"\nConfusion Matrix")
    print(metrics["confusion_matrix"])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    mode_label = "DUMMY" if DUMMY_MODE else "REAL"
    print(f"\n{'='*50}")
    print(f"EVALUATING SAVED MODELS ON {mode_label} DATASET")
    print(f"{'='*50}\n")

    # Locate saved models (fold checkpoints + standalone best_model)
    fold_models = sorted(glob.glob("best_fold_*.pth"))
    if os.path.exists("best_model.pth"):
        fold_models.append("best_model.pth")

    if not fold_models:
        print("ERROR: No saved models found (best_fold_*.pth or best_model.pth).")
        print("Run  python train.py  or  python rerun_pipeline.py  first.")
        sys.exit(1)

    print(f"Found {len(fold_models)} saved model(s): "
          f"{', '.join(os.path.basename(m) for m in fold_models)}\n")

    # Prepare dataset
    prepare_data()

    sequences, labels = load_dataset_json("sequence_dataset.json")

    total = len(labels)
    deaths = sum(labels)
    print(f"Dataset : {total} sequences  |  {deaths} deaths  |  {total - deaths} alive\n")

    # Evaluate each fold model
    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []

    for model_path in fold_models:
        fold_name = os.path.basename(model_path).replace(".pth", "")

        checkpoint = torch.load(model_path, map_location=DEVICE)

        # Support both old (raw state_dict) and new (dict with vocab) format
        if isinstance(checkpoint, dict) and "vocab" in checkpoint:
            vocab = checkpoint["vocab"]
            state_dict = checkpoint["model_state_dict"]
        else:
            # Legacy checkpoint: rebuild vocab from dataset
            vocab = build_vocab(sequences)
            state_dict = checkpoint

        dataset = MedicalDataset(sequences, labels, vocab)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        model = MedicalTransformer(len(vocab)).to(DEVICE)
        model.load_state_dict(state_dict)

        probs, true_labels = evaluate_model(model, loader)
        metrics = compute_metrics(probs, true_labels)

        print_metrics(metrics, header=f"{fold_name.upper()} RESULTS")

        all_aucs.append(metrics["auroc"])
        all_f1s.append(metrics["f1"])
        all_precisions.append(metrics["precision"])
        all_recalls.append(metrics["recall"])

    # Summary
    if len(fold_models) > 1:
        print(f"\n\n{'='*50}")
        print(f"ENSEMBLE SUMMARY ({len(fold_models)} models)")
        print("=" * 50)
        print(f"Mean AUROC     : {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
        print(f"Mean F1        : {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
        print(f"Mean Precision : {np.mean(all_precisions):.4f}")
        print(f"Mean Recall    : {np.mean(all_recalls):.4f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
