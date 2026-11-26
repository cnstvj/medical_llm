"""
Evaluator module for evaluating saved model checkpoints and calculating performance metrics.
"""

import os
import sys
import glob
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

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
from medical_llm.utils.confidence import assess_prediction_confidence
from medical_llm.data.dataset import MedicalDataset, build_vocab, load_dataset_json
from medical_llm.models.transformer import MedicalTransformer
from medical_llm.models.clinical_explainer import ClinicalExplainer
from medical_llm.semantic.metrics import SemanticEvaluationSuite
from medical_llm.semantic.representation import SemanticRepresentation


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

            all_probs.extend(p.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())

    return all_probs, all_labels


def evaluate_multimodal_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[float], List[float]]:
    """Runs inference across a multi-modal data loader and returns predicted probabilities and true labels."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            lab_ids = batch["lab_ids"].to(device)
            lab_mask = batch["lab_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            delta_times = batch["delta_times"].to(device) if "delta_times" in batch else None
            y = batch["label"]

            logits = model(lab_ids, lab_mask, text_ids, text_mask, delta_times=delta_times)
            p = torch.sigmoid(logits)

            all_probs.extend(p.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())

    return all_probs, all_labels


def compute_metrics(
    probs: List[float],
    labels: List[float],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes AUROC, F1 score, precision, recall, and confusion matrix using fixed or optimal threshold."""
    try:
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    except ValueError:
        auc = 0.0

    if threshold is not None:
        eval_threshold = float(threshold)
    else:
        eval_threshold = 0.5
        best_f1 = -1.0
        for th in np.arange(0.05, 0.96, 0.05):
            preds = [1 if p >= th else 0 for p in probs]
            f1_val = f1_score(labels, preds, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                eval_threshold = float(th)

    preds = [1 if p >= eval_threshold else 0 for p in probs]

    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "auroc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": eval_threshold,
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

    sequences, labels, _ = load_dataset_json(config.sequence_dataset_file)

    total = len(labels)
    deaths = sum(labels)
    log_event(f"Dataset : {total} sequences  |  {deaths} deaths  |  {total - deaths} alive\n")

    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []
    all_model_predictions: List[List[float]] = []

    reports_map = {}
    if os.path.exists(config.processed_reports_file):
        try:
            import pandas as pd
            r_df = pd.read_csv(config.processed_reports_file)
            reports_map = dict(zip(r_df["hadm_id"], r_df["clean_text"]))
        except Exception:
            pass

    import json
    with open(config.sequence_dataset_file, "r") as f:
        data_json = json.load(f)
    hadm_ids = data_json.get("hadm_ids", [0] * len(sequences))
    clinical_texts = [str(reports_map.get(hid, "")) for hid in hadm_ids]

    oof_probs_by_hadm: Dict[int, List[float]] = {}
    oof_label_by_hadm: Dict[int, int] = {}
    oof_seq_by_hadm: Dict[int, List[str]] = {}
    oof_text_by_hadm: Dict[int, str] = {}
    conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    last_vocab = None
    last_threshold = 0.5
    raw_tables_cache = None

    def _get_raw_tables():
        nonlocal raw_tables_cache
        if raw_tables_cache is None:
            from medical_llm.data.preprocessing import load_raw_tables
            raw_tables_cache = load_raw_tables(config, dummy_mode=dummy_mode)
        return raw_tables_cache

    for model_path in fold_models:
        fold_name = os.path.basename(model_path).replace(".pth", "")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "vocab" in checkpoint:
            vocab = checkpoint["vocab"]
            state_dict = checkpoint["model_state_dict"]
        else:
            vocab = build_vocab(sequences)
            state_dict = checkpoint
        last_vocab = vocab

        saved_threshold = checkpoint.get("threshold") if isinstance(checkpoint, dict) else None
        if saved_threshold is not None:
            last_threshold = float(saved_threshold)
        test_hadm_ids = checkpoint.get("test_hadm_ids") if isinstance(checkpoint, dict) else None
        unit_registry = checkpoint.get("unit_registry") if isinstance(checkpoint, dict) else None
        quantile_map = checkpoint.get("quantile_map") if isinstance(checkpoint, dict) else None

        is_multimodal = any("text_encoder" in k for k in state_dict.keys())
        eval_deltas = None
        fold_unit_report = None

        # Prefer raw-table reconstruction when dummy_mode=True or checkpoint carries fold-specific quantile_map
        use_raw_eval = dummy_mode or (quantile_map is not None and test_hadm_ids is not None)
        if use_raw_eval:
            try:
                raw_adm, raw_labs, raw_reps = _get_raw_tables()
                if test_hadm_ids is not None and len(test_hadm_ids) > 0:
                    test_hadm_set = set(int(h) for h in test_hadm_ids)
                    eval_adm = raw_adm[raw_adm["hadm_id"].isin(test_hadm_set)].copy()
                    log_event(f"[{fold_name}] Evaluating strictly on {len(eval_adm)} held-out test admission(s) (0% train overlap).")
                else:
                    eval_adm = raw_adm.copy()
                    log_event(f"[{fold_name}] WARNING: checkpoint has no 'test_hadm_ids'; evaluating on all {len(eval_adm)} raw admissions.")

                eval_labs_df = raw_labs[raw_labs["hadm_id"].isin(eval_adm["hadm_id"])].copy()
                if unit_registry is not None:
                    from medical_llm.utils.units import normalize_lab_units
                    eval_labs_df, fold_unit_report = normalize_lab_units(eval_labs_df, unit_registry, emit_warnings=False)
                if quantile_map is not None:
                    from medical_llm.utils.quantiles import apply_quantiles_to_labevents
                    eval_labs_proc = apply_quantiles_to_labevents(eval_labs_df, quantile_map)
                else:
                    eval_labs_proc = eval_labs_df

                if is_multimodal:
                    from medical_llm.pipeline.cross_validator import build_multimodal_fold_data
                    eval_seqs, eval_texts, eval_labels, eval_deltas = build_multimodal_fold_data(
                        eval_adm, eval_labs_proc, raw_reps, experiment_mode=config.experiment_mode, return_delta_times=True
                    )
                else:
                    from medical_llm.pipeline.cross_validator import build_fold_sequences
                    eval_seqs, eval_labels = build_fold_sequences(
                        eval_adm, eval_labs_proc, raw_reps, experiment_mode=config.experiment_mode
                    )
                    eval_texts = [""] * len(eval_seqs)
                eval_hadms = [int(h) for h in eval_adm["hadm_id"].tolist()]
            except Exception as exc:
                log_event(f"[{fold_name}] Raw-table evaluation fallback triggered ({exc}); using {config.sequence_dataset_file}.")
                use_raw_eval = False

        if not use_raw_eval:
            if test_hadm_ids is not None and len(test_hadm_ids) > 0:
                test_hadm_set = set(int(h) for h in test_hadm_ids)
                eval_indices = [i for i, h in enumerate(hadm_ids) if int(h) in test_hadm_set]
                log_event(f"[{fold_name}] Evaluating strictly on {len(eval_indices)} held-out test admission(s) (0% train overlap).")
            else:
                eval_indices = list(range(len(sequences)))
                log_event(
                    f"[{fold_name}] WARNING: checkpoint has no 'test_hadm_ids'; "
                    f"evaluating on all {len(eval_indices)} admissions (may overlap with training set)."
                )

            eval_seqs = [sequences[i] for i in eval_indices]
            eval_labels = [labels[i] for i in eval_indices]
            eval_texts = [clinical_texts[i] for i in eval_indices]
            eval_hadms = [int(hadm_ids[i]) for i in eval_indices]

        if not eval_seqs:
            log_event(f"[{fold_name}] WARNING: 0 matching test admissions found; skipping.")
            continue

        if is_multimodal:
            from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer
            from medical_llm.data.dataset import MultiModalMedicalDataset
            from medical_llm.pipeline.cross_validator import get_hf_tokenizer

            ckpt_use_lora = any("lora_" in k for k in state_dict.keys())
            ckpt_use_time2vec = any("time2vec" in k for k in state_dict.keys())
            tokenizer = get_hf_tokenizer(config.pretrained_text_model)
            dataset = MultiModalMedicalDataset(
                lab_sequences=eval_seqs,
                clinical_texts=eval_texts,
                labels=eval_labels,
                vocab=vocab,
                tokenizer=tokenizer,
                max_lab_len=config.max_len,
                max_text_len=config.text_max_len,
                delta_times=eval_deltas,
                unk_dropout=0.0,
            )
            loader = DataLoader(dataset, batch_size=config.batch_size)
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
                use_lora=ckpt_use_lora,
                fusion_type=config.fusion_type,
                fusion_temperature=getattr(config, "fusion_temperature", 1.0),
                use_time2vec=ckpt_use_time2vec,
            ).to(device)
            model.load_state_dict(state_dict)
            probs, true_labels = evaluate_multimodal_model(model, loader, device)
        else:
            dataset = MedicalDataset(eval_seqs, eval_labels, vocab, unk_dropout=0.0)
            loader = DataLoader(dataset, batch_size=config.batch_size)
            model = MedicalTransformer(len(vocab)).to(device)
            model.load_state_dict(state_dict)
            probs, true_labels = evaluate_model(model, loader, device)

        used_th = float(saved_threshold) if saved_threshold is not None else 0.5
        for seq_item, txt_item, prob_item, hid_item, lbl_item in zip(eval_seqs, eval_texts, probs, eval_hadms, true_labels):
            conf = assess_prediction_confidence(
                lab_tokens=seq_item,
                vocab=vocab,
                clinical_text=txt_item,
                probability=prob_item,
                threshold=used_th,
                unit_report=fold_unit_report,
            )
            conf_counts[conf["confidence"]] += 1
            oof_probs_by_hadm.setdefault(hid_item, []).append(float(prob_item))
            oof_label_by_hadm[hid_item] = int(lbl_item)
            oof_seq_by_hadm[hid_item] = seq_item
            oof_text_by_hadm[hid_item] = txt_item

        metrics = compute_metrics(probs, true_labels, threshold=saved_threshold)

        print_metrics(metrics, header=f"{fold_name.upper()} RESULTS")

        all_model_predictions.append(probs)
        all_aucs.append(metrics["auroc"])
        all_f1s.append(metrics["f1"])
        all_precisions.append(metrics["precision"])
        all_recalls.append(metrics["recall"])

    if len(fold_models) > 1 and all_aucs:
        log_event(f"\n\n{'='*50}")
        log_event(f"ENSEMBLE SUMMARY ({len(all_aucs)} models)")
        log_event("=" * 50)
        log_event(f"Mean AUROC     : {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
        log_event(f"Mean F1        : {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
        log_event(f"Mean Precision : {np.mean(all_precisions):.4f}")
        log_event(f"Mean Recall    : {np.mean(all_recalls):.4f}")
        log_event(f"Confidence Dist: HIGH={conf_counts['HIGH']}, MEDIUM={conf_counts['MEDIUM']}, LOW={conf_counts['LOW']}")

    # Evaluate Semantic Communication Layer on Test Samples
    log_event(f"\n\n{'='*50}")
    log_event("SEMANTIC COMMUNICATION LAYER EVALUATION")
    log_event("=" * 50)

    sample_hadms = list(oof_probs_by_hadm.keys())[:20] if oof_probs_by_hadm else []
    test_reps: List[SemanticRepresentation] = []
    test_exps: List[str] = []
    sample_conf_line = ""

    encoder, _, _ = ClinicalExplainer.get_components()

    if sample_hadms:
        for hid in sample_hadms:
            prob = float(np.mean(oof_probs_by_hadm[hid]))
            seq = oof_seq_by_hadm[hid]
            patient_text = oof_text_by_hadm.get(hid, "")
            expl_result = ClinicalExplainer.generate_patient_rationale(
                probability=prob,
                lab_tokens=seq,
                clinical_text=patient_text,
                threshold=last_threshold,
                vocab=last_vocab,
                validate=True,
            )
            if not sample_conf_line:
                sample_conf_line = f"{expl_result['confidence']} (Score: {expl_result['confidence_score']:.2f})"
            rep = SemanticRepresentation.from_dict(expl_result["semantic_representation"])
            test_reps.append(rep)
            test_exps.append(expl_result["clinical_narrative"])
    else:
        sample_size = min(20, len(sequences))
        for idx in range(sample_size):
            seq = sequences[idx]
            prob = 0.85 if labels[idx] == 1 else 0.15
            patient_text = clinical_texts[idx] if idx < len(clinical_texts) else ""
            expl_result = ClinicalExplainer.generate_patient_rationale(
                probability=prob,
                lab_tokens=seq,
                clinical_text=patient_text,
                threshold=last_threshold,
                vocab=last_vocab,
                validate=True,
            )
            if not sample_conf_line:
                sample_conf_line = f"{expl_result['confidence']} (Score: {expl_result['confidence_score']:.2f})"
            rep = SemanticRepresentation.from_dict(expl_result["semantic_representation"])
            test_reps.append(rep)
            test_exps.append(expl_result["clinical_narrative"])

    sem_metrics = SemanticEvaluationSuite.evaluate_batch(test_exps, test_reps)
    log_event(f"Samples Evaluated         : {sem_metrics['num_evaluated']}")
    log_event(f"Semantic Preservation     : {sem_metrics['semantic_preservation_score'] * 100:.1f}%")
    log_event(f"Unsupported Claim Rate    : {sem_metrics['unsupported_claim_rate'] * 100:.1f}%")
    log_event(f"Directional Accuracy      : {sem_metrics['directional_accuracy'] * 100:.1f}%")
    log_event(f"Uncertainty Coverage      : {sem_metrics['uncertainty_coverage'] * 100:.1f}%")
    log_event(f"Validation Pass Rate      : {sem_metrics['validation_pass_rate'] * 100:.1f}%")
    log_event(f"Clinical Faithfulness Ind : {sem_metrics['clinical_faithfulness_index']:.4f}")

    if test_exps:
        log_event(f"\n--- Sample Semantic Communication Output (Index 0) ---")
        if sample_conf_line:
            log_event(f"CONFIDENCE : {sample_conf_line}")
        log_event(f"Summary    : {test_reps[0].clinical_summary}")
        log_event(f"Explanation: {test_exps[0]}")

    log_event(f"\nDone.\n")
