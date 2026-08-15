"""
Cross-validation engine for Stratified 5-Fold training and evaluation.
"""

from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from medical_llm.config import DEFAULT_CONFIG, PipelineConfig
from medical_llm.data.dataset import MedicalDataset, build_vocab, load_dataset_json
from medical_llm.models.transformer import MedicalTransformer


def evaluate_fold(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """Evaluates fold model and returns metrics (auroc, f1, precision, recall, threshold, confusion matrix)."""
    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)

            logits = model(x, mask)
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
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

    return auc, best_f1, precision, recall, best_threshold, cm


def train_fold(
    train_sequences: List[List[str]],
    train_labels: List[int],
    test_sequences: List[List[str]],
    test_labels: List[int],
    fold: int,
    vocab: dict,
    device: torch.device,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[float, float, float, float]:
    """Trains a single fold model with cosine annealing and early stopping."""
    train_dataset = MedicalDataset(train_sequences, train_labels, vocab, max_len=config.max_len)
    test_dataset = MedicalDataset(test_sequences, test_labels, vocab, max_len=config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = MedicalTransformer(
        len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        val_auc, _, _, _, _, _ = evaluate_fold(model, test_loader, device)

        print(f"Fold {fold} Epoch {epoch+1:2d} Loss={total_loss:.4f} Val_AUROC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                },
                f"best_fold_{fold}.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping (Fold {fold})")
            break

    checkpoint = torch.load(f"best_fold_{fold}.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    auc, f1, precision, recall, threshold, cm = evaluate_fold(model, test_loader, device)

    print("\n" + "=" * 40)
    print(f"FOLD {fold} RESULTS")
    print("=" * 40)
    print(f"AUROC     : {auc:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Threshold : {threshold:.2f}")
    print("\nConfusion Matrix")
    print(cm)

    return auc, f1, precision, recall


def run_cross_validation(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Runs Stratified 5-Fold Cross Validation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences, labels = load_dataset_json(config.sequence_dataset_file)

    outer_train_sequences, outer_val_sequences, outer_train_labels, outer_val_labels = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=config.random_seed
    )

    print(
        f"Total samples={len(sequences)}, "
        f"outer train={len(outer_train_sequences)}, "
        f"outer validation={len(outer_val_sequences)}"
    )

    vocab = build_vocab(outer_train_sequences)
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)

    aucs, f1s, precisions, recalls = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(outer_train_sequences, outer_train_labels), start=1
    ):
        train_seqs = [outer_train_sequences[i] for i in train_idx]
        train_lbls = [outer_train_labels[i] for i in train_idx]
        test_seqs = [outer_train_sequences[i] for i in test_idx]
        test_lbls = [outer_train_labels[i] for i in test_idx]

        auc, f1, precision, recall = train_fold(
            train_seqs, train_lbls, test_seqs, test_lbls, fold, vocab, device, config=config
        )

        aucs.append(auc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    print("\n")
    print("=" * 50)
    print(f"FINAL CROSS-VALIDATION RESULTS - {config.experiment_mode}")
    print("=" * 50)
    print(f"Mean AUROC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Mean F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Mean Precision : {np.mean(precisions):.4f}")
    print(f"Mean Recall    : {np.mean(recalls):.4f}")
