"""
Tabular and Sequence Feature Extraction for Clinical Gradient Boosted Trees (XGBoost / LightGBM)
and Multimodal Ensemble.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def extract_tabular_features(
    admissions_df: pd.DataFrame,
    labevents_df: pd.DataFrame,
    reports_df: Optional[pd.DataFrame] = None,
    itemids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts rich clinical tabular summary features per admission:
    - Demographics: Age, Gender, Admission Type
    - Lab Aggregates: Mean, Std, Min, Max, Count, Delta (first to last) per lab item
    - Clinical Note metadata: Word count, presence of report
    """
    feature_rows = []
    labels = []
    
    # Pre-index labevents by hadm_id for speed
    labs_grouped = dict(tuple(labevents_df.groupby("hadm_id")))
    reports_map = {}
    if reports_df is not None and "hadm_id" in reports_df.columns:
        reports_map = dict(zip(reports_df["hadm_id"], reports_df.get("clean_text", [""] * len(reports_df))))

    if itemids is not None:
        all_itemids = list(itemids)
    else:
        # Rank by true frequency in training labevents
        item_counts = labevents_df["itemid"].dropna().value_counts()
        all_itemids = list(item_counts.index[:30])

    feature_names = ["age", "gender_is_m", "is_emergency"]
    for iid in all_itemids:
        feature_names.extend([f"lab_{iid}_min", f"lab_{iid}_max", f"lab_{iid}_mean", f"lab_{iid}_count", f"lab_{iid}_delta"])
    feature_names.extend(["note_char_count", "note_word_count"])

    for _, adm in admissions_df.iterrows():
        hadm_id = adm["hadm_id"]
        row_feats = []

        # Demographics
        age = float(adm.get("age", 50.0))
        gender_m = 1.0 if str(adm.get("gender", "")).upper() == "M" else 0.0
        is_emerg = 1.0 if "EMERGENCY" in str(adm.get("admission_type", "")).upper() else 0.0
        row_feats.extend([age, gender_m, is_emerg])

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

        # Note stats
        note = str(reports_map.get(hadm_id, ""))
        note_chars = float(len(note))
        note_words = float(len(note.split()))
        row_feats.extend([note_chars, note_words])

        feature_rows.append(row_feats)
        labels.append(int(adm.get("target", 0)))

    X = np.array(feature_rows, dtype=np.float32)
    # Fill nan with 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, dtype=np.int64)

    return X, y, feature_names
