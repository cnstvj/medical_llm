"""
Trainer module for training MedicalTransformer and MultiModalMedicalTransformer models.
"""

import math
import os
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from medical_llm.config import DEFAULT_CONFIG, PipelineConfig
from medical_llm.utils.logging import log_event
from medical_llm.data.dataset import (
    MedicalDataset,
    MultiModalMedicalDataset,
    get_dataloaders,
    train_test_split_by_subject,
    train_test_split_custom,
    build_vocab,
)
from medical_llm.models.transformer import MedicalTransformer
from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer


def prepare_data(dummy_mode: bool = False, config: PipelineConfig = DEFAULT_CONFIG, force_rebuild: bool = False) -> None:
    """Ensures required data artifacts are generated via direct Python APIs without brittle subprocesses."""
    from medical_llm.data.preprocessing import run_preprocessing
    from medical_llm.data.sequence_builder import run_sequence_builder
    from synthetic.generate_dummy_data import generate_dummy_datasets

    if dummy_mode:
        adm_dummy = config.get_csv_name(config.admissions_csv, dummy_mode=True)
        if not os.path.exists(adm_dummy):
            log_event("[DUMMY MODE] Generating synthetic dummy CSV files...")
            generate_dummy_datasets(verbose=True)

    if force_rebuild or not os.path.exists(config.sequence_dataset_file):
        log_event("Running preprocessing...")
        run_preprocessing(config=config, dummy_mode=dummy_mode)
        log_event("Running sequence builder...")
        run_sequence_builder(config=config)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    """Evaluates standard model performance and returns AUROC, F1, precision, and recall metrics."""
    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    preds = [1 if p > 0.5 else 0 for p in probs]

    try:
        auc = roc_auc_score(labels, probs)
        if math.isnan(auc):
            auc = 0.0
    except Exception:
        auc = 0.0

    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    return auc, f1, prec, rec


def evaluate_multimodal(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    """Evaluates multi-modal model performance and returns AUROC, F1, precision, and recall metrics."""
    model.eval()
    probs = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            lab_ids = batch["lab_ids"].to(device)
            lab_mask = batch["lab_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            y = batch["label"].to(device)

            logits = model(lab_ids, lab_mask, text_ids, text_mask)
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    preds = [1 if p > 0.5 else 0 for p in probs]

    try:
        auc = roc_auc_score(labels, probs)
        if math.isnan(auc):
            auc = 0.0
    except Exception:
        auc = 0.0

    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    return auc, f1, prec, rec


def train(dummy_mode: bool = False, config: PipelineConfig = DEFAULT_CONFIG, force_rebuild: bool = False) -> None:
    """Main training loop for MedicalTransformer or MultiModalMedicalTransformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_label = "DUMMY" if dummy_mode else "REAL"
    arch_label = "MULTI-MODAL" if config.use_multimodal_nlp else "TIME-SERIES"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_event("\n" + "=" * 60)
    log_event(f"[{timestamp}] EVENT: TRAIN PIPELINE RUN ({mode_label} DATASET | {arch_label})")
    log_event("=" * 60 + "\n")

    prepare_data(dummy_mode=dummy_mode, config=config, force_rebuild=force_rebuild)

    if config.use_multimodal_nlp:
        # Load reports mapping
        reports_map = {}
        if os.path.exists(config.processed_reports_file):
            import pandas as pd
            r_df = pd.read_csv(config.processed_reports_file)
            reports_map = dict(zip(r_df["hadm_id"], r_df["clean_text"]))

        with open(config.sequence_dataset_file, "r") as f:
            data = json.load(f)

        sequences = data["sequences"]
        labels = data["labels"]
        subject_ids = data.get("subject_ids", [i for i in range(len(labels))])
        hadm_ids = data.get("hadm_ids", [0] * len(labels))
        clinical_texts = [str(reports_map.get(hid, "")) for hid in hadm_ids]

        # Patient-level split with synchronized clinical text alignment
        seq_train, seq_temp, y_train, y_temp, sid_train, sid_temp, train_texts, temp_texts = train_test_split_by_subject(
            sequences, labels, subject_ids, test_size=0.2, random_state=config.random_seed, texts=clinical_texts
        )
        seq_val, seq_test, y_val, y_test, _, _, val_texts, test_texts = train_test_split_by_subject(
            seq_temp, y_temp, sid_temp, test_size=0.5, random_state=config.random_seed, texts=temp_texts
        )


        from medical_llm.pipeline.cross_validator import get_hf_tokenizer
        tokenizer = get_hf_tokenizer(config.pretrained_text_model)

        vocab = build_vocab(seq_train)

        train_dataset = MultiModalMedicalDataset(
            seq_train, train_texts, y_train, vocab, tokenizer=tokenizer,
            max_lab_len=config.max_len, max_text_len=config.text_max_len
        )
        val_dataset = MultiModalMedicalDataset(
            seq_val, val_texts, y_val, vocab, tokenizer=tokenizer,
            max_lab_len=config.max_len, max_text_len=config.text_max_len
        )
        test_dataset = MultiModalMedicalDataset(
            seq_test, test_texts, y_test, vocab, tokenizer=tokenizer,
            max_lab_len=config.max_len, max_text_len=config.text_max_len
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        model = MultiModalMedicalTransformer(
            vocab_size=len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_lab_len=config.max_len,
            pretrained_text_model=config.pretrained_text_model,
            freeze_text_backbone=config.freeze_text_backbone,
            fusion_type=config.fusion_type,
        ).to(device)

        eval_fn = evaluate_multimodal
    else:
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            config.sequence_dataset_file, batch_size=config.batch_size, max_len=config.max_len
        )
        y_train = train_loader.dataset.labels

        model = MedicalTransformer(
            len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_len,
        ).to(device)

        eval_fn = evaluate

    num_positive = sum(y_train)
    num_negative = len(y_train) - num_positive
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    log_event(f"Training samples: {len(y_train)} (Pos: {num_positive}, Neg: {num_negative})")
    log_event(f"Class imbalance weight (pos_weight): {pos_weight.item():.4f}\n")

    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        if config.use_multimodal_nlp:
            for batch in train_loader:
                lab_ids = batch["lab_ids"].to(device)
                lab_mask = batch["lab_mask"].to(device)
                text_ids = batch["text_ids"].to(device)
                text_mask = batch["text_mask"].to(device)
                y = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(lab_ids, lab_mask, text_ids, text_mask)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
        else:
            for x, mask, y in train_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)

                optimizer.zero_grad()
                logits = model(x, mask)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        scheduler.step()

        val_auc, val_f1, val_prec, val_rec = eval_fn(model, val_loader, device)
        log_event(f"Epoch {epoch+1:2d}/{config.epochs}: Loss={total_loss:.4f} AUROC={val_auc:.4f} F1={val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "is_multimodal": config.use_multimodal_nlp,
                },
                "best_model.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            log_event(f"Early stopping at epoch {epoch+1}")
            break

    checkpoint = torch.load("best_model.pth", map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    test_auc, test_f1, test_prec, test_rec = eval_fn(model, test_loader, device)
    log_event(f"\nTEST RESULT: AUROC={test_auc:.4f} F1={test_f1:.4f} Precision={test_prec:.4f} Recall={test_rec:.4f}")
    log_event("Model saved to best_model.pth\n")
