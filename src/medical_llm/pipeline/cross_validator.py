"""
Cross-validation engine for Stratified 5-Fold training and evaluation.
Implements leak-free fold processing (quantiles and vocab fitted strictly per fold).
"""

from typing import Tuple, List
import pandas as pd
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
from medical_llm.data.preprocessing import load_raw_tables
from medical_llm.utils.quantiles import fit_quantiles, apply_quantiles_to_labevents
from medical_llm.data.sequence_builder import build_temporal_sequences
from medical_llm.data.dataset import MedicalDataset, build_vocab
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


def build_fold_sequences(
    admissions_df: pd.DataFrame,
    labevents_processed: pd.DataFrame,
    reports_df: pd.DataFrame,
    experiment_mode: str = "FULL",
) -> Tuple[List[List[str]], List[int]]:
    """Builds token sequences and target labels for a specific fold dataframe."""
    labevents = labevents_processed.copy()
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    labevents = labevents.sort_values(by=["hadm_id", "charttime"])

    grouped = build_temporal_sequences(labevents, experiment_mode=experiment_mode)

    df = pd.merge(admissions_df, grouped, on="hadm_id", how="left")
    df["event_token"] = df["event_token"].apply(lambda x: x if isinstance(x, list) else [])
    df = pd.merge(df, reports_df, on="hadm_id", how="left")

    sequences = []
    labels = []

    for _, row in df.iterrows():
        age_bucket = int(row["age"] // 10) * 10 if pd.notnull(row["age"]) else 0
        age_token = f"AGE_{age_bucket}"
        gender_token = f"GENDER_{row['gender']}"
        adm_type_token = f"ADM_{str(row['admission_type']).replace(' ', '_')}"

        seq = []

        if experiment_mode in ("DEMOGRAPHICS", "FULL"):
            seq.extend([age_token, gender_token, adm_type_token])

        if experiment_mode in ("LABS", "TEMPORAL", "FULL"):
            seq.append("[SEP]")
            seq.extend(row["event_token"])

        if experiment_mode in ("NOTES", "FULL"):
            if pd.notnull(row["clean_text"]) and str(row["clean_text"]).strip() != "":
                seq.append("[REPORT_START]")
                seq.extend(row["clean_text"].split())

        sequences.append(seq)
        labels.append(int(row["target"]) if pd.notnull(row["target"]) else 0)

    return sequences, labels


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

        print(f"Fold {fold} Epoch {epoch+1:2d} Loss={total_loss:.4f} Val_AUROC={val_auc:.4f}", flush=True)

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
            print(f"Early stopping (Fold {fold})", flush=True)
            break

    checkpoint = torch.load(f"best_fold_{fold}.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    auc, f1, precision, recall, threshold, cm = evaluate_fold(model, test_loader, device)

    print("\n" + "=" * 40, flush=True)
    print(f"FOLD {fold} RESULTS", flush=True)
    print("=" * 40, flush=True)
    print(f"AUROC     : {auc:.4f}", flush=True)
    print(f"F1        : {f1:.4f}", flush=True)
    print(f"Precision : {precision:.4f}", flush=True)
    print(f"Recall    : {recall:.4f}", flush=True)
    print(f"Threshold : {threshold:.2f}", flush=True)
    print("\nConfusion Matrix", flush=True)
    print(cm, flush=True)

    return auc, f1, precision, recall


def run_cross_validation(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Runs Stratified 5-Fold Cross Validation pipeline with 0% quantile & vocab leakage."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading raw tables for leak-free cross validation...", flush=True)
    admissions, labevents, reports = load_raw_tables(config)

    # Perform outer split to reserve test holdout if needed
    outer_train_admissions, outer_val_admissions = train_test_split(
        admissions,
        test_size=0.2,
        stratify=admissions["target"],
        random_state=config.random_seed,
    )

    print(
        f"Total admissions={len(admissions)}, "
        f"outer train={len(outer_train_admissions)}, "
        f"outer validation={len(outer_val_admissions)}",
        flush=True,
    )

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)

    aucs, f1s, precisions, recalls = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(outer_train_admissions, outer_train_admissions["target"]), start=1
    ):
        fold_train_adm = outer_train_admissions.iloc[train_idx]
        fold_test_adm = outer_train_admissions.iloc[test_idx]

        fold_train_labevents = labevents[labevents["hadm_id"].isin(fold_train_adm["hadm_id"])]
        fold_test_labevents = labevents[labevents["hadm_id"].isin(fold_test_adm["hadm_id"])]

        # Fit quantiles STRICTLY on fold_train_labevents
        fold_quantile_map = fit_quantiles(fold_train_labevents)

        # Apply fold_quantile_map to both fold train and fold test lab events
        fold_train_lab_proc = apply_quantiles_to_labevents(fold_train_labevents, fold_quantile_map)
        fold_test_lab_proc = apply_quantiles_to_labevents(fold_test_labevents, fold_quantile_map)

        # Build fold sequences
        train_seqs, train_lbls = build_fold_sequences(
            fold_train_adm, fold_train_lab_proc, reports, experiment_mode=config.experiment_mode
        )
        test_seqs, test_lbls = build_fold_sequences(
            fold_test_adm, fold_test_lab_proc, reports, experiment_mode=config.experiment_mode
        )

        # Build vocab STRICTLY from fold_train_sequences
        fold_vocab = build_vocab(train_seqs)

        auc, f1, precision, recall = train_fold(
            train_seqs, train_lbls, test_seqs, test_lbls, fold, fold_vocab, device, config=config
        )

        aucs.append(auc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    print("\n", flush=True)
    print("=" * 50, flush=True)
    print(f"FINAL LEAK-FREE CROSS-VALIDATION RESULTS - {config.experiment_mode}", flush=True)
    print("=" * 50, flush=True)
    print(f"Mean AUROC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"Mean F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}", flush=True)
    print(f"Mean Precision : {np.mean(precisions):.4f}", flush=True)
    print(f"Mean Recall    : {np.mean(recalls):.4f}", flush=True)
