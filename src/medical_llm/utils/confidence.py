"""
Prediction Confidence & Out-of-Distribution (OOD) Assessment Utilities.

Always produces a CONFIDENCE flag ("HIGH", "MEDIUM", or "LOW") alongside
prediction outputs by evaluating:
1. Known-pattern coverage (in-vocabulary token ratio, excluding special tokens).
2. Unit integrity (unconverted unit mismatches vs successfully converted units).
3. Modality completeness (lab trajectory presence + clinical text presence).
4. Decision margin (optional distance from classification threshold).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch

from medical_llm.utils.units import UnitConversionReport


SPECIAL_TOKENS = {"[PAD]", "[CLS]", "[SEP]", "[REPORT_START]", "[MASK]"}
BACKOFF_TOKENS = {"LAB_UNKNOWN_NORMAL", "LAB_UNKNOWN_ABNORMAL"}


def _is_demographic_or_time_token(token: str) -> bool:
    return (
        token.startswith("AGE_")
        or token.startswith("GENDER_")
        or token.startswith("ADM_")
        or token.startswith("TIME_")
    )


def assess_prediction_confidence(
    lab_tokens: List[str],
    vocab: Optional[Dict[str, int]] = None,
    clinical_text: str = "",
    probability: Optional[float] = None,
    threshold: float = 0.5,
    unit_report: Optional[UnitConversionReport] = None,
    extreme_value_rate: float = 0.0,
    require_text: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates input familiarity and unit integrity to produce a mandatory
    CONFIDENCE flag ("HIGH", "MEDIUM", or "LOW") and diagnostic reasons.
    """
    reasons: List[str] = []
    warnings_list: List[str] = []

    content_tokens = [t for t in (lab_tokens or []) if t not in SPECIAL_TOKENS]
    lab_event_tokens = [
        t for t in content_tokens
        if t.startswith("LAB_") or t == "[UNK]"
    ]

    total_content = len(content_tokens)
    total_lab_events = len(lab_event_tokens)

    unk_count = 0
    backoff_count = 0
    known_count = 0

    for tok in content_tokens:
        if tok == "[UNK]":
            unk_count += 1
        elif tok in BACKOFF_TOKENS:
            backoff_count += 1
        elif vocab is not None:
            if tok in vocab and vocab[tok] != vocab.get("[UNK]", -1):
                known_count += 1
            else:
                unk_count += 1
        else:
            # Without vocab, treat non-[UNK] tokens as known
            known_count += 1

    # Both explicit [UNK] and unseen-itemid backoff tokens represent unforeseen lab patterns
    unforeseen_count = unk_count + backoff_count
    known_ratio = (known_count / total_content) if total_content > 0 else 0.0
    lab_known_ratio = (
        sum(1 for t in lab_event_tokens if t not in BACKOFF_TOKENS and t != "[UNK]" and (vocab is None or t in vocab))
        / total_lab_events
        if total_lab_events > 0
        else 0.0
    )

    unconverted_mismatches = unit_report.unconverted_count if unit_report is not None else 0
    converted_units = unit_report.converted_count if unit_report is not None else 0
    if unit_report is not None and unit_report.warnings:
        warnings_list.extend(unit_report.warnings)

    has_text = bool(clinical_text and str(clinical_text).strip() and str(clinical_text).strip().lower() != "nan")

    # Compute confidence score in [0.0, 1.0]
    score = 1.0
    if total_content == 0:
        score = 0.1
        reasons.append("Empty input sequence.")
    else:
        # Penalize unforeseen tokens
        unforeseen_frac = 1.0 - known_ratio
        score -= 0.70 * unforeseen_frac
        if unforeseen_count > 0:
            reasons.append(
                f"{unforeseen_count}/{total_content} token(s) unforeseen or mapped to fallback "
                f"(known pattern coverage: {known_ratio * 100:.1f}%)."
            )

    if total_lab_events == 0:
        score -= 0.25
        reasons.append("No lab measurement events present in sequence.")
    elif total_lab_events < 3:
        score -= 0.10
        reasons.append(f"Sparse lab trajectory ({total_lab_events} lab event(s)).")

    if require_text and not has_text:
        score -= 0.10
        reasons.append("No clinical text/note provided.")

    if unconverted_mismatches > 0:
        mismatch_frac = unconverted_mismatches / max(total_lab_events, 1)
        score -= min(0.50, 0.25 + 0.50 * mismatch_frac)
        reasons.append(f"{unconverted_mismatches} lab event(s) had unconvertible unit mismatches.")

    if converted_units > 0:
        reasons.append(f"{converted_units} lab event(s) automatically converted to training units.")

    if extreme_value_rate > 0.30:
        score -= 0.15
        reasons.append(f"High rate of extreme out-of-range lab values ({extreme_value_rate * 100:.1f}%).")

    if probability is not None and abs(float(probability) - float(threshold)) < 0.08:
        score -= 0.05
        reasons.append(f"Predicted probability ({probability:.3f}) is near decision threshold ({threshold:.2f}).")

    score = float(max(0.0, min(1.0, score)))

    # Determine discrete CONFIDENCE flag
    mismatch_rate = unconverted_mismatches / max(total_lab_events, 1)
    is_low = (
        total_content == 0
        or known_ratio < 0.50
        or (total_lab_events > 0 and lab_known_ratio < 0.50)
        or (unconverted_mismatches >= 2)
        or (unconverted_mismatches > 0 and mismatch_rate >= 0.20)
        or (total_lab_events == 0 and not has_text)
    )

    is_medium = (not is_low) and (
        known_ratio < 0.85
        or (total_lab_events > 0 and lab_known_ratio < 0.85)
        or unconverted_mismatches > 0
        or total_lab_events < 3
        or (require_text and not has_text)
        or extreme_value_rate > 0.30
    )

    if is_low:
        flag = "LOW"
    elif is_medium:
        flag = "MEDIUM"
    else:
        flag = "HIGH"
        if not reasons:
            reasons.append("All input patterns and units match training distribution.")

    return {
        "confidence": flag,
        "confidence_score": round(score, 4),
        "known_token_ratio": round(known_ratio, 4),
        "lab_known_ratio": round(lab_known_ratio, 4),
        "unk_token_count": int(unk_count),
        "backoff_token_count": int(backoff_count),
        "total_content_tokens": int(total_content),
        "total_lab_events": int(total_lab_events),
        "unconverted_unit_mismatches": int(unconverted_mismatches),
        "converted_units_count": int(converted_units),
        "reasons": reasons,
        "warnings": warnings_list,
    }


def predict_with_confidence(
    model: torch.nn.Module,
    lab_sequences: List[List[str]],
    clinical_texts: Optional[List[str]],
    vocab: Dict[str, int],
    tokenizer: Any = None,
    device: Optional[torch.device] = None,
    max_lab_len: int = 512,
    max_text_len: int = 256,
    threshold: float = 0.5,
    delta_times: Optional[List[List[float]]] = None,
    unit_reports: Optional[List[Optional[UnitConversionReport]]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs model inference on raw sequences/texts and ALWAYS returns prediction
    probability, binary decision, and CONFIDENCE flag + diagnostics per sample.
    """
    from medical_llm.data.dataset import (
        MedicalDataset,
        MultiModalMedicalDataset,
        _truncate_preserving_demographics_and_recent,
    )
    from torch.utils.data import DataLoader

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    is_multimodal = clinical_texts is not None and hasattr(model, "text_encoder")

    dummy_labels = [0] * len(lab_sequences)
    if is_multimodal:
        ds = MultiModalMedicalDataset(
            lab_sequences=lab_sequences,
            clinical_texts=clinical_texts,
            labels=dummy_labels,
            vocab=vocab,
            tokenizer=tokenizer,
            max_lab_len=max_lab_len,
            max_text_len=max_text_len,
            delta_times=delta_times,
            unk_dropout=0.0,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        probs: List[float] = []
        with torch.no_grad():
            for batch in loader:
                l_i = batch["lab_ids"].to(device)
                l_m = batch["lab_mask"].to(device)
                t_i = batch["text_ids"].to(device)
                t_m = batch["text_mask"].to(device)
                dt = batch["delta_times"].to(device) if "delta_times" in batch else None
                logits = model(l_i, l_m, t_i, t_m, delta_times=dt)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    else:
        ds = MedicalDataset(lab_sequences, dummy_labels, vocab, max_len=max_lab_len, unk_dropout=0.0)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        probs = []
        with torch.no_grad():
            for x, mask, _ in loader:
                x, mask = x.to(device), mask.to(device)
                logits = model(x, mask)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    results: List[Dict[str, Any]] = []
    for idx, seq in enumerate(lab_sequences):
        prob = float(probs[idx])
        txt = clinical_texts[idx] if clinical_texts is not None and idx < len(clinical_texts) else ""
        u_rep = unit_reports[idx] if unit_reports is not None and idx < len(unit_reports) else None
        effective_seq, _ = _truncate_preserving_demographics_and_recent(seq, max_lab_len - 1)

        conf = assess_prediction_confidence(
            lab_tokens=effective_seq,
            vocab=vocab,
            clinical_text=txt,
            probability=prob,
            threshold=threshold,
            unit_report=u_rep,
            require_text=is_multimodal,
        )
        results.append({
            "probability": prob,
            "prediction": int(prob >= threshold),
            "threshold": float(threshold),
            "confidence": conf["confidence"],
            "confidence_score": conf["confidence_score"],
            "confidence_reasons": conf["reasons"],
            "warnings": conf["warnings"],
            "ood_metrics": conf,
        })

    return results


def predict_from_raw_labs(
    model: torch.nn.Module,
    admissions_df: Any,
    labevents_df: Any,
    vocab: Dict[str, int],
    quantile_map: Dict[Any, Any],
    unit_registry: Optional[Dict[int, Dict[str, str]]] = None,
    reports_df: Any = None,
    tokenizer: Any = None,
    device: Optional[torch.device] = None,
    max_lab_len: int = 512,
    max_text_len: int = 256,
    threshold: float = 0.5,
    emit_warnings: bool = True,
) -> List[Dict[str, Any]]:
    """
    End-to-end inference helper from raw labevents:
    1. Normalizes/converts lab units against `unit_registry` (or warns & masks unconvertible units).
    2. Discretizes values using `quantile_map`.
    3. Builds demographic + temporal lab sequences and clinical texts.
    4. Runs `predict_with_confidence` and attaches per-admission unit conversion reports.
    """
    import pandas as pd
    from medical_llm.utils.units import normalize_lab_units
    from medical_llm.utils.quantiles import apply_quantiles_to_labevents
    from medical_llm.data.sequence_builder import build_temporal_sequences
    from medical_llm.pipeline.cross_validator import synthesize_clinical_text

    adm = admissions_df.copy()
    if "target" not in adm.columns:
        adm["target"] = 0

    labs = labevents_df.copy() if labevents_df is not None else pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum"])
    if "charttime" in labs.columns and not labs.empty:
        labs["charttime"] = pd.to_datetime(labs["charttime"])
        labs = labs.sort_values(by=["hadm_id", "charttime"])

    # Normalize units per admission so each admission receives its own UnitConversionReport
    per_adm_reports: List[Optional[UnitConversionReport]] = []
    normalized_chunks = []
    for hid in adm["hadm_id"].tolist():
        adm_labs = labs[labs["hadm_id"] == hid] if not labs.empty else labs
        if unit_registry is not None and not adm_labs.empty:
            norm_chunk, rep = normalize_lab_units(adm_labs, unit_registry, emit_warnings=emit_warnings)
            normalized_chunks.append(norm_chunk)
            per_adm_reports.append(rep)
        else:
            if not adm_labs.empty:
                normalized_chunks.append(adm_labs)
            per_adm_reports.append(None)

    norm_labs = pd.concat(normalized_chunks, ignore_index=True) if normalized_chunks else labs
    proc_labs = apply_quantiles_to_labevents(norm_labs, quantile_map) if not norm_labs.empty else norm_labs
    if not proc_labs.empty:
        grouped = build_temporal_sequences(proc_labs, experiment_mode="FULL")
    else:
        grouped = pd.DataFrame({"hadm_id": [], "event_token": [], "delta_times": []})

    merged = pd.merge(adm, grouped, on="hadm_id", how="left")
    if reports_df is not None and not reports_df.empty and "hadm_id" in reports_df.columns:
        merged = pd.merge(merged, reports_df, on="hadm_id", how="left")

    lab_sequences: List[List[str]] = []
    clinical_texts: List[str] = []
    delta_times_list: List[List[float]] = []

    for _, row in merged.iterrows():
        age_val = row.get("age", 50)
        age_bucket = int(age_val // 10) * 10 if pd.notnull(age_val) else 0
        gender_val = row.get("gender", "M")
        adm_type_val = row.get("admission_type", "EMERGENCY")
        seq = [f"AGE_{age_bucket}", f"GENDER_{gender_val}", f"ADM_{str(adm_type_val).replace(' ', '_')}", "[SEP]"]

        ev_tokens = row["event_token"] if isinstance(row.get("event_token"), list) else []
        ev_deltas = row["delta_times"] if isinstance(row.get("delta_times"), list) else [0.0] * len(ev_tokens)
        seq.extend(ev_tokens)
        deltas = [0.0, 0.0, 0.0, 0.0] + ev_deltas

        lab_sequences.append(seq)
        delta_times_list.append(deltas)
        clinical_texts.append(
            synthesize_clinical_text(
                row.get("age"),
                row.get("gender"),
                row.get("admission_type"),
                row.get("diagnosis", ""),
                row.get("clean_text", ""),
            )
        )

    is_multimodal = hasattr(model, "text_encoder")
    return predict_with_confidence(
        model=model,
        lab_sequences=lab_sequences,
        clinical_texts=clinical_texts if is_multimodal else None,
        vocab=vocab,
        tokenizer=tokenizer,
        device=device,
        max_lab_len=max_lab_len,
        max_text_len=max_text_len,
        threshold=threshold,
        delta_times=delta_times_list,
        unit_reports=per_adm_reports,
    )
