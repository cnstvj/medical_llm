"""
Audit script for checking saved checkpoints, prediction distributions, text encoder verification,
and detailed Fold 3 analysis.
"""

import sys
import os
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import PipelineConfig
from medical_llm.data.preprocessing import load_raw_tables
from medical_llm.utils.quantiles import fit_quantiles, apply_quantiles_to_labevents
from medical_llm.data.dataset import MultiModalMedicalDataset
from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer, ClinicalTextEncoder
from medical_llm.pipeline.cross_validator import get_hf_tokenizer, build_multimodal_fold_data, evaluate_multimodal_fold

def audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PipelineConfig()
    
    print("=" * 60)
    print("1. VERIFYING TEXT ENCODER")
    print("=" * 60)
    text_encoder = ClinicalTextEncoder(
        pretrained_model_name=config.pretrained_text_model,
        embedding_dim=config.text_embedding_dim,
        proj_dim=config.d_model,
        freeze_backbone=config.freeze_text_backbone,
    )
    print(f"Pretrained Model Name: {config.pretrained_text_model}")
    print(f"Text Encoder Class   : {text_encoder.__class__.__name__}")
    print(f"Backbone Submodule   : {text_encoder.bert.__class__.__name__} ({text_encoder.bert.__class__.__module__})")
    print(f"HF Loaded (use_hf)   : {text_encoder.use_hf}")
    if text_encoder.use_hf and hasattr(text_encoder.bert, "config"):
        print(f"Backbone Config Type : {type(text_encoder.bert.config)}")
        print(f"Hidden Size          : {text_encoder.bert.config.hidden_size}")
        print(f"Num Hidden Layers    : {text_encoder.bert.config.num_hidden_layers}")
        print(f"Num Attention Heads  : {text_encoder.bert.config.num_attention_heads}")
        print(f"Vocab Size           : {text_encoder.bert.config.vocab_size}")
        print(f"Pretrained Weights   : SUCCESS (Loaded from HuggingFace cache/hub)")
    else:
        print(f"Pretrained Weights   : FALLBACK (nn.Embedding fallback was used)")
    print()

    print("=" * 60)
    print("2. EVALUATING SAVED CHECKPOINTS ACROSS FOLDS")
    print("=" * 60)
    admissions, labevents, reports = load_raw_tables(config, dummy_mode=True)
    
    subject_ids = admissions["subject_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.random_seed)
    outer_train_idx, outer_val_idx = next(gss.split(admissions, admissions["target"], groups=subject_ids))
    outer_train_admissions = admissions.iloc[outer_train_idx].copy()
    
    inner_subject_ids = outer_train_admissions["subject_id"].values
    inner_labels = outer_train_admissions["target"].values
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    
    tokenizer = get_hf_tokenizer(config.pretrained_text_model)

    for fold, (train_idx, test_idx) in enumerate(
        sgkf.split(outer_train_admissions, inner_labels, groups=inner_subject_ids), start=1
    ):
        ckpt_path = os.path.join(BASE_DIR, f"best_fold_{fold}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found!")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        vocab = checkpoint["vocab"]
        is_multimodal = checkpoint.get("is_multimodal", True)

        fold_train_adm = outer_train_admissions.iloc[train_idx]
        fold_test_adm = outer_train_admissions.iloc[test_idx]

        fold_train_labevents = labevents[labevents["hadm_id"].isin(fold_train_adm["hadm_id"])]
        fold_test_labevents = labevents[labevents["hadm_id"].isin(fold_test_adm["hadm_id"])]

        fold_quantile_map = fit_quantiles(fold_train_labevents)
        fold_test_lab_proc = apply_quantiles_to_labevents(fold_test_labevents, fold_quantile_map)

        test_labs, test_texts, test_lbls = build_multimodal_fold_data(
            fold_test_adm, fold_test_lab_proc, reports, experiment_mode=config.experiment_mode
        )

        test_dataset = MultiModalMedicalDataset(
            test_labs, test_texts, test_lbls, vocab=vocab, tokenizer=tokenizer,
            max_lab_len=config.max_len, max_text_len=config.text_max_len
        )
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

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
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Gather predictions
        probs = []
        labels = []
        with torch.no_grad():
            for batch in test_loader:
                lab_ids = batch["lab_ids"].to(device)
                lab_mask = batch["lab_mask"].to(device)
                text_ids = batch["text_ids"].to(device)
                text_mask = batch["text_mask"].to(device)
                y = batch["label"]
                logits = model(lab_ids, lab_mask, text_ids, text_mask)
                p = torch.sigmoid(logits)
                probs.extend(p.cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())

        probs_arr = np.array(probs)
        labels_arr = np.array(labels)
        
        auc = roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.0

        # Optimal threshold search (same as pipeline)
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in np.arange(0.05, 0.96, 0.05):
            preds_t = [1 if p > threshold else 0 for p in probs_arr]
            f1 = f1_score(labels_arr, preds_t, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        preds = np.array([1 if p > best_threshold else 0 for p in probs_arr])
        
        # Also compute at fixed threshold 0.5
        preds_fixed = np.array([1 if p > 0.5 else 0 for p in probs_arr])

        # Confusion matrix elements: TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(labels_arr, preds, labels=[0, 1]).ravel()
        pos_preds = int(preds.sum())
        
        tn_50, fp_50, fn_50, tp_50 = confusion_matrix(labels_arr, preds_fixed, labels=[0, 1]).ravel()
        pos_preds_50 = int(preds_fixed.sum())

        print(f"--- FOLD {fold} PREDICTION AUDIT ---")
        print(f"Total Test Samples: {len(labels_arr)} (Deaths/Positive={int(labels_arr.sum())}, Survivors/Negative={int((labels_arr == 0).sum())})")
        print(f"AUROC: {auc:.4f} | Optimal Threshold: {best_threshold:.2f}")
        print(f"At Optimal Threshold ({best_threshold:.2f}):")
        print(f"  Positive Predictions: {pos_preds}")
        print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
        print(f"  Precision: {precision_score(labels_arr, preds, zero_division=0):.4f} | Recall: {recall_score(labels_arr, preds, zero_division=0):.4f} | F1: {best_f1:.4f}")
        print(f"At Fixed Threshold (0.50):")
        print(f"  Positive Predictions: {pos_preds_50}")
        print(f"  TP: {tp_50} | FP: {fp_50} | TN: {tn_50} | FN: {fn_50}")
        
        print(f"Detailed Sample Probabilities and True Labels:")
        for idx, (p_val, y_true) in enumerate(zip(probs_arr, labels_arr)):
            print(f"  Sample {idx+1}: True Label={int(y_true)}, Pred Prob={p_val:.4f}, Pred@{best_threshold:.2f}={int(p_val > best_threshold)}, Pred@0.5={int(p_val > 0.5)}")

        if fold == 3:
            print("\n*** IN-DEPTH FOLD 3 ANALYSIS ***")
            print("Pairwise ranking of (Positive, Negative) pairs:")
            pos_indices = np.where(labels_arr == 1)[0]
            neg_indices = np.where(labels_arr == 0)[0]
            pair_count = 0
            correct_pairs = 0
            for pi in pos_indices:
                for ni in neg_indices:
                    pair_count += 1
                    status = "CORRECT" if probs_arr[pi] > probs_arr[ni] else ("TIE" if probs_arr[pi] == probs_arr[ni] else "INVERTED")
                    if probs_arr[pi] > probs_arr[ni]:
                        correct_pairs += 1
                    elif probs_arr[pi] == probs_arr[ni]:
                        correct_pairs += 0.5
                    print(f"  Pair (Pos Sample {pi+1} prob={probs_arr[pi]:.4f} vs Neg Sample {ni+1} prob={probs_arr[ni]:.4f}) -> {status}")
            print(f"Total Pairs: {pair_count}, Concordant Pairs: {correct_pairs}")
            print(f"Calculated AUROC = {correct_pairs} / {pair_count} = {correct_pairs / pair_count:.4f}")
        print()

if __name__ == "__main__":
    audit()
