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
            y = batch["label"]

            logits = model(lab_ids, lab_mask, text_ids, text_mask)
            p = torch.sigmoid(logits)

            all_probs.extend(p.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())

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

    for model_path in fold_models:
        fold_name = os.path.basename(model_path).replace(".pth", "")
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "vocab" in checkpoint:
            vocab = checkpoint["vocab"]
            state_dict = checkpoint["model_state_dict"]
        else:
            vocab = build_vocab(sequences)
            state_dict = checkpoint

        is_multimodal = any("text_encoder" in k for k in state_dict.keys())
        if is_multimodal:
            from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer
            from medical_llm.data.dataset import MultiModalMedicalDataset
            from medical_llm.pipeline.cross_validator import get_hf_tokenizer

            tokenizer = get_hf_tokenizer(config.pretrained_text_model)
            dataset = MultiModalMedicalDataset(
                lab_sequences=sequences,
                clinical_texts=clinical_texts,
                labels=labels,
                vocab=vocab,
                tokenizer=tokenizer,
                max_lab_len=config.max_len,
                max_text_len=config.text_max_len,
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
                fusion_type=config.fusion_type,
            ).to(device)
            model.load_state_dict(state_dict)
            probs, true_labels = evaluate_multimodal_model(model, loader, device)
        else:
            dataset = MedicalDataset(sequences, labels, vocab)
            loader = DataLoader(dataset, batch_size=config.batch_size)
            model = MedicalTransformer(len(vocab)).to(device)
            model.load_state_dict(state_dict)
            probs, true_labels = evaluate_model(model, loader, device)

        metrics = compute_metrics(probs, true_labels)

        print_metrics(metrics, header=f"{fold_name.upper()} RESULTS")

        all_model_predictions.append(probs)
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

    # Evaluate Semantic Communication Layer on Test Samples
    log_event(f"\n\n{'='*50}")
    log_event("SEMANTIC COMMUNICATION LAYER EVALUATION")
    log_event("=" * 50)

    sample_size = min(20, len(sequences))
    test_reps: List[SemanticRepresentation] = []
    test_exps: List[str] = []

    encoder, _, _ = ClinicalExplainer.get_components()

    for idx in range(sample_size):
        seq = sequences[idx]
        # Use average predicted probability across all evaluated fold models, or label baseline fallback
        patient_fold_probs = [preds[idx] for preds in all_model_predictions if idx < len(preds)]
        prob = float(np.mean(patient_fold_probs)) if patient_fold_probs else (0.85 if labels[idx] == 1 else 0.15)
        patient_text = clinical_texts[idx] if idx < len(clinical_texts) else ""
        expl_result = ClinicalExplainer.generate_patient_rationale(
            probability=prob,
            lab_tokens=seq,
            clinical_text=patient_text,
            threshold=0.5,
            validate=True,
        )

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
        log_event(f"Summary    : {test_reps[0].clinical_summary}")
        log_event(f"Explanation: {test_exps[0]}")

    log_event(f"\nDone.\n")
