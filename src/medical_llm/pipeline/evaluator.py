"""
Evaluator module for evaluating saved model checkpoints and calculating performance metrics.
"""

import os
import sys
import glob
from datetime import datetime
from typing import List, Tuple, Dict, Any

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

from medical_llm.config import DEFAULT_CONFIG, PipelineConfig
from medical_llm.utils.logging import log_event
from medical_llm.data.dataset import MedicalDataset, build_vocab, load_dataset_json
from medical_llm.models.transformer import MedicalTransformer


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[float], List[float]]:
    """Runs inference across a data loader and returns predicted probabilities and true labels."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)

            logits = model(x, mask)
            p = torch.sigmoid(logits)

            all_probs.extend(p.cpu().numpy())
            all_labels.extend(y.numpy())

    return all_probs, all_labels


def compute_metrics(probs: List[float], labels: List[float]) -> Dict[str, Any]:
    """Computes AUROC, optimal threshold F1 score, precision, recall, and confusion matrix."""
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


def print_metrics(metrics: Dict[str, Any], header: str = "", log_file: str = DEFAULT_CONFIG.log_file) -> None:
    """Logs evaluation metrics to stdout and log file."""
    if header:
        log_event(f"\n{'='*40}", log_file=log_file)
        log_event(header, log_file=log_file)
        log_event("=" * 40, log_file=log_file)

    log_event(f"AUROC     : {metrics['auroc']:.4f}", log_file=log_file)
    log_event(f"F1        : {metrics['f1']:.4f}", log_file=log_file)
    log_event(f"Precision : {metrics['precision']:.4f}", log_file=log_file)
    log_event(f"Recall    : {metrics['recall']:.4f}", log_file=log_file)
    log_event(f"Threshold : {metrics['threshold']:.2f}", log_file=log_file)
    log_event(f"\nConfusion Matrix", log_file=log_file)
    log_event(str(metrics["confusion_matrix"]), log_file=log_file)


def run_test_evaluation(dummy_mode: bool = False, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Runs evaluation over all available saved model fold checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_label = "DUMMY" if dummy_mode else "REAL"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_event(f"\n{'='*60}")
    log_event(f"[{timestamp}] EVENT: TEST EVALUATION RUN ({mode_label} DATASET)")
    log_event(f"{'='*60}\n")

    fold_models = sorted(glob.glob("best_fold_*.pth"))
    if os.path.exists("best_model.pth"):
        fold_models.append("best_model.pth")

    if not fold_models:
        log_event("ERROR: No saved models found (best_fold_*.pth or best_model.pth).")
        log_event("Run  python train.py  or  python rerun_pipeline.py  first.")
        sys.exit(1)

    log_event(f"Found {len(fold_models)} saved model(s): "
              f"{', '.join(os.path.basename(m) for m in fold_models)}\n")

    sequences, labels = load_dataset_json(config.sequence_dataset_file)

    total = len(labels)
    deaths = sum(labels)
    log_event(f"Dataset : {total} sequences  |  {deaths} deaths  |  {total - deaths} alive\n")

    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []

    for model_path in fold_models:
        fold_name = os.path.basename(model_path).replace(".pth", "")
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "vocab" in checkpoint:
            vocab = checkpoint["vocab"]
            state_dict = checkpoint["model_state_dict"]
        else:
            vocab = build_vocab(sequences)
            state_dict = checkpoint

        dataset = MedicalDataset(sequences, labels, vocab)
        loader = DataLoader(dataset, batch_size=config.batch_size)

        model = MedicalTransformer(len(vocab)).to(device)
        model.load_state_dict(state_dict)

        probs, true_labels = evaluate_model(model, loader, device)
        metrics = compute_metrics(probs, true_labels)

        print_metrics(metrics, header=f"{fold_name.upper()} RESULTS")

        all_aucs.append(metrics["auroc"])
        all_f1s.append(metrics["f1"])
        all_precisions.append(metrics["precision"])
        all_recalls.append(metrics["recall"])

    if len(fold_models) > 1:
        log_event(f"\n\n{'='*50}")
        log_event(f"ENSEMBLE SUMMARY ({len(fold_models)} models)")
        log_event("=" * 50)
        log_event(f"Mean AUROC     : {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
        log_event(f"Mean F1        : {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
        log_event(f"Mean Precision : {np.mean(all_precisions):.4f}")
        log_event(f"Mean Recall    : {np.mean(all_recalls):.4f}")

    log_event(f"\nDone.\n")
