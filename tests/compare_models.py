"""
Comprehensive side-by-side comparison of Unimodal vs Multimodal models under identical conditions:
- Same random seed (42)
- Same GroupShuffleSplit outer split and StratifiedGroupKFold 5-fold CV splits
- Same test samples and patient groups
- Same evaluation thresholds (fixed 0.50 and optimal threshold)
- Full confusion matrix breakdowns (TP, FP, TN, FN, positive predictions)
"""

import sys
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import PipelineConfig
from medical_llm.data.preprocessing import load_raw_tables
from medical_llm.utils.quantiles import fit_quantiles, apply_quantiles_to_labevents
from medical_llm.data.dataset import MedicalDataset, MultiModalMedicalDataset, build_vocab
from medical_llm.models.transformer import MedicalTransformer
from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer
from medical_llm.pipeline.cross_validator import (
    get_hf_tokenizer,
    build_fold_sequences,
    build_multimodal_fold_data,
    train_fold,
    train_multimodal_fold,
)


def evaluate_probs(y_true, y_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_probs)
    
    preds = np.array([1 if p > threshold else 0 for p in y_probs])
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    pos_preds = int(preds.sum())
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    return {
        "auroc": auc,
        "auprc": auprc,
        "brier": brier,
        "pos_preds": pos_preds,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": threshold,
    }


def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PipelineConfig()
    config.random_seed = 42

    print("=" * 80)
    print("IDENTICAL CONDITIONS BENCHMARK: UNIMODAL VS MULTIMODAL (5-FOLD CV)")
    print("=" * 80)
    print(f"Device: {device} | Seed: {config.random_seed} | Splits: {config.n_splits}")
    print(f"Dataset: Dummy MIMIC-III (50 admissions, 27 patients)")
    
    admissions, labevents, reports = load_raw_tables(config, dummy_mode=True)
    
    # Outer split (Patient-level)
    subject_ids = admissions["subject_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.random_seed)
    outer_train_idx, outer_val_idx = next(gss.split(admissions, admissions["target"], groups=subject_ids))
    outer_train_admissions = admissions.iloc[outer_train_idx].copy()
    
    inner_subject_ids = outer_train_admissions["subject_id"].values
    inner_labels = outer_train_admissions["target"].values
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    
    tokenizer = get_hf_tokenizer(config.pretrained_text_model)
    
    unimodal_results = []
    multimodal_results = []
    
    splits = list(sgkf.split(outer_train_admissions, inner_labels, groups=inner_subject_ids))
    
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n>>> PROCESSING FOLD {fold}/5 <<<")
        fold_train_adm = outer_train_admissions.iloc[train_idx]
        fold_test_adm = outer_train_admissions.iloc[test_idx]

        fold_train_labevents = labevents[labevents["hadm_id"].isin(fold_train_adm["hadm_id"])]
        fold_test_labevents = labevents[labevents["hadm_id"].isin(fold_test_adm["hadm_id"])]

        # Fit quantiles STRICTLY on fold train
        fold_quantile_map = fit_quantiles(fold_train_labevents)
        fold_train_lab_proc = apply_quantiles_to_labevents(fold_train_labevents, fold_quantile_map)
        fold_test_lab_proc = apply_quantiles_to_labevents(fold_test_labevents, fold_quantile_map)

        # ----------------- UNIMODAL -----------------
        torch.manual_seed(config.random_seed + fold)
        np.random.seed(config.random_seed + fold)
        
        train_seqs, train_lbls = build_fold_sequences(fold_train_adm, fold_train_lab_proc, reports, experiment_mode=config.experiment_mode)
        test_seqs, test_lbls = build_fold_sequences(fold_test_adm, fold_test_lab_proc, reports, experiment_mode=config.experiment_mode)
        fold_vocab_uni = build_vocab(train_seqs)

        train_ds_uni = MedicalDataset(train_seqs, train_lbls, fold_vocab_uni, max_len=config.max_len)
        test_ds_uni = MedicalDataset(test_seqs, test_lbls, fold_vocab_uni, max_len=config.max_len)
        train_ld_uni = DataLoader(train_ds_uni, batch_size=config.batch_size, shuffle=True)
        test_ld_uni = DataLoader(test_ds_uni, batch_size=config.batch_size, shuffle=False)

        model_uni = MedicalTransformer(
            vocab_size=len(fold_vocab_uni),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_len,
        ).to(device)

        pos_w = torch.tensor([(len(train_lbls) - sum(train_lbls)) / max(sum(train_lbls), 1)], dtype=torch.float32, device=device)
        crit_uni = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt_uni = torch.optim.AdamW(model_uni.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        for ep in range(config.max_epochs):
            model_uni.train()
            for x, mask, y in train_ld_uni:
                opt_uni.zero_grad()
                out = model_uni(x.to(device), mask.to(device))
                loss = crit_uni(out, y.to(device))
                loss.backward()
                opt_uni.step()

        model_uni.eval()
        probs_uni = []
        with torch.no_grad():
            for x, mask, y in test_ld_uni:
                p = torch.sigmoid(model_uni(x.to(device), mask.to(device)))
                probs_uni.extend(p.cpu().numpy().tolist())

        # ----------------- MULTIMODAL -----------------
        torch.manual_seed(config.random_seed + fold)
        np.random.seed(config.random_seed + fold)

        train_labs, train_texts, train_lbls_multi = build_multimodal_fold_data(fold_train_adm, fold_train_lab_proc, reports, experiment_mode=config.experiment_mode)
        test_labs, test_texts, test_lbls_multi = build_multimodal_fold_data(fold_test_adm, fold_test_lab_proc, reports, experiment_mode=config.experiment_mode)
        fold_vocab_multi = build_vocab(train_labs)

        train_ds_multi = MultiModalMedicalDataset(train_labs, train_texts, train_lbls_multi, vocab=fold_vocab_multi, tokenizer=tokenizer, max_lab_len=config.max_len, max_text_len=config.text_max_len)
        test_ds_multi = MultiModalMedicalDataset(test_labs, test_texts, test_lbls_multi, vocab=fold_vocab_multi, tokenizer=tokenizer, max_lab_len=config.max_len, max_text_len=config.text_max_len)
        train_ld_multi = DataLoader(train_ds_multi, batch_size=config.batch_size, shuffle=True)
        test_ld_multi = DataLoader(test_ds_multi, batch_size=config.batch_size, shuffle=False)

        model_multi = MultiModalMedicalTransformer(
            vocab_size=len(fold_vocab_multi),
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

        crit_multi = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt_multi = torch.optim.AdamW(model_multi.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        for ep in range(config.max_epochs):
            model_multi.train()
            for batch in train_ld_multi:
                opt_multi.zero_grad()
                out = model_multi(
                    batch["lab_ids"].to(device),
                    batch["lab_mask"].to(device),
                    batch["text_ids"].to(device),
                    batch["text_mask"].to(device),
                )
                loss = crit_multi(out, batch["label"].to(device))
                loss.backward()
                opt_multi.step()

        model_multi.eval()
        probs_multi = []
        with torch.no_grad():
            for batch in test_ld_multi:
                out = model_multi(
                    batch["lab_ids"].to(device),
                    batch["lab_mask"].to(device),
                    batch["text_ids"].to(device),
                    batch["text_mask"].to(device),
                )
                p = torch.sigmoid(out)
                probs_multi.extend(p.cpu().numpy().tolist())

        # Evaluate both at Threshold = 0.50
        uni_res_50 = evaluate_probs(test_lbls, probs_uni, threshold=0.50)
        multi_res_50 = evaluate_probs(test_lbls_multi, probs_multi, threshold=0.50)
        
        # Also find best threshold for each
        best_th_uni, best_f1_uni = 0.5, 0.0
        best_th_multi, best_f1_multi = 0.5, 0.0
        for th in np.arange(0.05, 0.96, 0.05):
            f1_u = f1_score(test_lbls, [1 if p > th else 0 for p in probs_uni], zero_division=0)
            if f1_u > best_f1_uni:
                best_f1_uni, best_th_uni = f1_u, th
            f1_m = f1_score(test_lbls_multi, [1 if p > th else 0 for p in probs_multi], zero_division=0)
            if f1_m > best_f1_multi:
                best_f1_multi, best_th_multi = f1_m, th

        uni_res_opt = evaluate_probs(test_lbls, probs_uni, threshold=best_th_uni)
        multi_res_opt = evaluate_probs(test_lbls_multi, probs_multi, threshold=best_th_multi)

        unimodal_results.append({"th50": uni_res_50, "opt": uni_res_opt, "probs": probs_uni, "labels": test_lbls})
        multimodal_results.append({"th50": multi_res_50, "opt": multi_res_opt, "probs": probs_multi, "labels": test_lbls_multi})

        print(f"Fold {fold} (Test N={len(test_lbls)}, Deaths={sum(test_lbls)}, Survivors={len(test_lbls)-sum(test_lbls)}):")
        print(f"  Unimodal   -> AUROC={uni_res_50['auroc']:.4f} | AUPRC={uni_res_50['auprc']:.4f} | OptTh={best_th_uni:.2f} (F1={uni_res_opt['f1']:.4f}, TP={uni_res_opt['tp']}, FP={uni_res_opt['fp']}, TN={uni_res_opt['tn']}, FN={uni_res_opt['fn']})")
        print(f"  Multimodal -> AUROC={multi_res_50['auroc']:.4f} | AUPRC={multi_res_50['auprc']:.4f} | OptTh={best_th_multi:.2f} (F1={multi_res_opt['f1']:.4f}, TP={multi_res_opt['tp']}, FP={multi_res_opt['fp']}, TN={multi_res_opt['tn']}, FN={multi_res_opt['fn']})")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON UNDER IDENTICAL CONDITIONS (ACROSS 5 FOLDS)")
    print("=" * 80)
    
    uni_aurocs = [r["th50"]["auroc"] for r in unimodal_results]
    multi_aurocs = [r["th50"]["auroc"] for r in multimodal_results]
    uni_auprcs = [r["th50"]["auprc"] for r in unimodal_results]
    multi_auprcs = [r["th50"]["auprc"] for r in multimodal_results]
    uni_briers = [r["th50"]["brier"] for r in unimodal_results]
    multi_briers = [r["th50"]["brier"] for r in multimodal_results]
    
    uni_f1_opt = [r["opt"]["f1"] for r in unimodal_results]
    multi_f1_opt = [r["opt"]["f1"] for r in multimodal_results]

    print(f"Metric                 | Unimodal (Labs Only)    | Multimodal (+ Bio_ClinicalBERT)")
    print(f"-----------------------+-------------------------+--------------------------------")
    print(f"Mean AUROC             | {np.mean(uni_aurocs):.4f} +/- {np.std(uni_aurocs):.4f}      | {np.mean(multi_aurocs):.4f} +/- {np.std(multi_aurocs):.4f}")
    print(f"Mean AUPRC             | {np.mean(uni_auprcs):.4f} +/- {np.std(uni_auprcs):.4f}      | {np.mean(multi_auprcs):.4f} +/- {np.std(multi_auprcs):.4f}")
    print(f"Mean Brier Score       | {np.mean(uni_briers):.4f} +/- {np.std(uni_briers):.4f}      | {np.mean(multi_briers):.4f} +/- {np.std(multi_briers):.4f}")
    print(f"Mean F1 (Optimal Th)   | {np.mean(uni_f1_opt):.4f} +/- {np.std(uni_f1_opt):.4f}      | {np.mean(multi_f1_opt):.4f} +/- {np.std(multi_f1_opt):.4f}")
    print(f"-----------------------+-------------------------+--------------------------------")
    print("\nFold-by-Fold AUROC Comparison:")
    for f in range(5):
        print(f"  Fold {f+1}: Unimodal AUROC = {uni_aurocs[f]:.4f} | Multimodal AUROC = {multi_aurocs[f]:.4f} | Delta = {multi_aurocs[f] - uni_aurocs[f]:+.4f}")

if __name__ == "__main__":
    run_comparison()
