"""
Tabular and Sequence Feature Extraction for Clinical Gradient Boosted Trees (XGBoost / LightGBM)
and Multimodal Ensemble.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


# Diagnosis keyword indicators for clinical feature encoding
DIAGNOSIS_INDICATORS = {
    "diag_sepsis": ["SEPSIS", "SEPTIC"],
    "diag_pneumonia": ["PNEUMONIA", "PNEUM"],
    "diag_heart_failure": ["HEART FAILURE", "CHF", "CONGESTIVE"],
    "diag_renal_disease": ["RENAL", "KIDNEY", "DIALYSIS"],
    "diag_liver_disease": ["LIVER", "HEPATIC", "HEPATITIS", "CIRRHOSIS"],
    "diag_respiratory": ["RESPIRATORY", "DYSPNEA", "SHORTNESS OF BREATH", "COPD", "HYPOXIA"],
    "diag_stroke": ["STROKE", "TIA", "INFARCT", "INTRACRANIAL", "HEMORRHAG"],
    "diag_cancer": ["CANCER", "CA/", "NEOPLASM", "MALIGNAN", "METASTA", "TUMOR", "LEUKEMIA", "LYMPHOMA"],
}


def extract_tabular_features(
    admissions_df: pd.DataFrame,
    labevents_df: pd.DataFrame,
    reports_df: Optional[pd.DataFrame] = None,
    itemids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts rich clinical tabular summary features per admission:
    - Demographics: Age, Gender, Admission Type
    - Admission Diagnosis Indicators: Sepsis, Pneumonia, Heart Failure, Renal/Liver Disease, etc.
    - Lab Aggregates: Mean, Min, Max, Count, Delta (first to last) per lab item

    Note: Raw clinical note length/word-count features are intentionally excluded
    to prevent note-presence mortality leakage.
    """
    feature_rows = []
    labels = []

    # Pre-index labevents by hadm_id for speed
    labs_grouped = dict(tuple(labevents_df.groupby("hadm_id")))

    if itemids is not None:
        all_itemids = list(itemids)
    else:
        # Rank by true frequency in training labevents
        item_counts = labevents_df["itemid"].dropna().value_counts()
        all_itemids = list(item_counts.index[:30])

    feature_names = ["age", "gender_is_m", "is_emergency"]
    feature_names.extend(list(DIAGNOSIS_INDICATORS.keys()))
    for iid in all_itemids:
        feature_names.extend([f"lab_{iid}_min", f"lab_{iid}_max", f"lab_{iid}_mean", f"lab_{iid}_count", f"lab_{iid}_delta"])

    for _, adm in admissions_df.iterrows():
        hadm_id = adm["hadm_id"]
        row_feats = []

        # Demographics
        age = float(adm.get("age", 50.0))
        gender_m = 1.0 if str(adm.get("gender", "")).upper() == "M" else 0.0
        is_emerg = 1.0 if "EMERGENCY" in str(adm.get("admission_type", "")).upper() else 0.0
        row_feats.extend([age, gender_m, is_emerg])

        # Diagnosis indicators
        diag_str = str(adm.get("diagnosis", "")).upper()
        for diag_key, keywords in DIAGNOSIS_INDICATORS.items():
            has_diag = 1.0 if any(kw in diag_str for kw in keywords) else 0.0
            row_feats.append(has_diag)

        # Labs
        hadm_labs = labs_grouped.get(hadm_id, None)
        for iid in all_itemids:
            if hadm_labs is not None:
                item_vals = hadm_labs[hadm_labs["itemid"] == iid]["valuenum"].dropna().values
                if len(item_vals) > 0:
                    row_feats.extend([
                        float(np.min(item_vals)),
                        float(np.max(item_vals)),
                        float(np.mean(item_vals)),
                        float(len(item_vals)),
                        float(item_vals[-1] - item_vals[0]),
                    ])
                else:
                    row_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                row_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        feature_rows.append(row_feats)
        labels.append(int(adm.get("target", 0)))

    X = np.array(feature_rows, dtype=np.float32)
    # Fill nan with 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, dtype=np.int64)

    return X, y, feature_names
