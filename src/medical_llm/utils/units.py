"""
Clinical Unit Normalization, Conversion, and Mismatch Warning Utilities.

Fits a canonical unit registry from training lab events, converts compatible
units (SI/metric and analyte-specific molar/mass conversions) before feeding
values to quantile discretization / transformer models, and warns on
unconvertible unit mismatches while preventing corrupted numeric values from
entering quantile buckets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd


# Canonical alias normalization (lowercase, stripped -> canonical representation)
_UNIT_ALIASES: Dict[str, str] = {
    "mg/dl": "mg/dL",
    "mg/l": "mg/L",
    "g/dl": "g/dL",
    "g/l": "g/L",
    "ug/ml": "ug/mL",
    "mcg/ml": "ug/mL",
    "ng/ml": "ng/mL",
    "pg/ml": "pg/mL",
    "meq/l": "mEq/L",
    "mmol/l": "mmol/L",
    "umol/l": "umol/L",
    "mcmol/l": "umol/L",
    "k/ul": "K/uL",
    "10^3/ul": "K/uL",
    "x10^3/ul": "K/uL",
    "m/ul": "m/uL",
    "10^6/ul": "m/uL",
    "x10^6/ul": "m/uL",
    "#/ul": "#/uL",
    "cells/ul": "#/uL",
    "#/hpf": "#/hpf",
    "iu/l": "IU/L",
    "u/l": "IU/L",
    "units": "units",
    "mm hg": "mm Hg",
    "mmhg": "mm Hg",
    "kpa": "kPa",
    "sec": "sec",
    "seconds": "sec",
    "second": "sec",
    "s": "sec",
    "min": "min",
    "mosm/kg": "mOsm/kg",
    "mosm/l": "mOsm/L",
    "uu/ml": "uIU/mL",
    "uiu/ml": "uIU/mL",
    "miu/ml": "mIU/mL",
    "mu/ml": "mIU/mL",
    "%": "%",
    "percent": "%",
    "ratio": "ratio",
    "fraction": "ratio",
    "fl": "fL",
    "pg": "pg",
    "eu/dl": "EU/dL",
    "degc": "C",
    "c": "C",
    "degf": "F",
    "f": "F",
}

# General linear conversions: (from_unit, to_unit) -> (scale, offset) where new = old * scale + offset
_GENERAL_CONVERSIONS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("g/dL", "g/L"): (10.0, 0.0),
    ("g/L", "g/dL"): (0.1, 0.0),
    ("mg/dL", "mg/L"): (10.0, 0.0),
    ("mg/L", "mg/dL"): (0.1, 0.0),
    ("mg/dL", "g/L"): (0.01, 0.0),
    ("g/L", "mg/dL"): (100.0, 0.0),
    ("mg/dL", "ug/mL"): (10.0, 0.0),
    ("ug/mL", "mg/dL"): (0.1, 0.0),
    ("ug/mL", "ng/mL"): (1000.0, 0.0),
    ("ng/mL", "ug/mL"): (0.001, 0.0),
    ("ng/mL", "pg/mL"): (1000.0, 0.0),
    ("pg/mL", "ng/mL"): (0.001, 0.0),
    ("mmol/L", "umol/L"): (1000.0, 0.0),
    ("umol/L", "mmol/L"): (0.001, 0.0),
    ("K/uL", "#/uL"): (1000.0, 0.0),
    ("#/uL", "K/uL"): (0.001, 0.0),
    ("K/uL", "m/uL"): (0.001, 0.0),
    ("m/uL", "K/uL"): (1000.0, 0.0),
    ("sec", "min"): (1.0 / 60.0, 0.0),
    ("min", "sec"): (60.0, 0.0),
    ("mm Hg", "kPa"): (0.133322, 0.0),
    ("kPa", "mm Hg"): (1.0 / 0.133322, 0.0),
    ("%", "ratio"): (0.01, 0.0),
    ("ratio", "%"): (100.0, 0.0),
    ("C", "F"): (1.8, 32.0),
    ("F", "C"): (5.0 / 9.0, -32.0 * 5.0 / 9.0),
}

# Analyte-specific molar <-> mass conversions (matched by lowercase label keyword)
# Each entry: keyword -> ((unit_a, unit_b), factor_a_to_b)
_ANALYTE_CONVERSIONS: List[Tuple[str, str, str, float]] = [
    ("glucose", "mg/dL", "mmol/L", 1.0 / 18.0156),
    ("creatinine", "mg/dL", "umol/L", 88.402),
    ("urea", "mg/dL", "mmol/L", 0.357),
    ("bun", "mg/dL", "mmol/L", 0.357),
    ("calcium", "mg/dL", "mmol/L", 0.2495),
    ("bilirubin", "mg/dL", "umol/L", 17.104),
    ("cholesterol", "mg/dL", "mmol/L", 0.02586),
    ("lactate", "mmol/L", "mg/dL", 9.008),
    ("hemoglobin", "g/dL", "mmol/L", 0.6206),
    ("sodium", "mEq/L", "mmol/L", 1.0),
    ("potassium", "mEq/L", "mmol/L", 1.0),
    ("chloride", "mEq/L", "mmol/L", 1.0),
    ("bicarbonate", "mEq/L", "mmol/L", 1.0),
]


def canonicalize_unit(unit: Any) -> str:
    """Normalizes a unit string into a canonical representation, or empty string if missing."""
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    raw = str(unit).strip()
    if raw == "" or raw.lower() == "nan":
        return ""
    return _UNIT_ALIASES.get(raw.lower(), raw)


@dataclass
class UnitConversionReport:
    """Summary of unit normalization, conversions, and unconvertible mismatches."""
    converted_count: int = 0
    unconverted_count: int = 0
    missing_unit_count: int = 0
    unseen_itemid_count: int = 0
    conversions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unseen_itemids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converted_count": self.converted_count,
            "unconverted_count": self.unconverted_count,
            "missing_unit_count": self.missing_unit_count,
            "unseen_itemid_count": self.unseen_itemid_count,
            "conversions": list(self.conversions),
            "warnings": list(self.warnings),
            "unseen_itemids": list(self.unseen_itemids),
        }


def fit_unit_registry(
    train_labevents: pd.DataFrame,
    d_labitems: Optional[pd.DataFrame] = None,
) -> Dict[int, Dict[str, str]]:
    """
    Fits a per-itemid canonical unit registry strictly from training labevents.
    Returns: {itemid: {"unit": canonical_unit, "label": label_str}}
    """
    registry: Dict[int, Dict[str, str]] = {}
    if train_labevents is None or train_labevents.empty or "itemid" not in train_labevents.columns:
        return registry

    label_map: Dict[int, str] = {}
    if "label" in train_labevents.columns:
        for iid, lbl in zip(train_labevents["itemid"], train_labevents["label"]):
            if pd.notnull(iid) and pd.notnull(lbl):
                label_map[int(iid)] = str(lbl)
    if d_labitems is not None and "itemid" in d_labitems.columns and "label" in d_labitems.columns:
        for iid, lbl in zip(d_labitems["itemid"], d_labitems["label"]):
            if pd.notnull(iid) and pd.notnull(lbl):
                label_map.setdefault(int(iid), str(lbl))

    has_uom = "valueuom" in train_labevents.columns
    for iid, group in train_labevents.groupby("itemid"):
        if pd.isna(iid):
            continue
        iid_int = int(iid)
        canonical_unit = ""
        if has_uom:
            units = [canonicalize_unit(u) for u in group["valueuom"].tolist()]
            non_empty = [u for u in units if u != ""]
            if non_empty:
                # Modal (most frequent) unit in training data
                counts = pd.Series(non_empty).value_counts()
                canonical_unit = str(counts.index[0])
        registry[iid_int] = {
            "unit": canonical_unit,
            "label": label_map.get(iid_int, ""),
        }

    return registry


def convert_unit_value(
    value: float,
    from_unit: str,
    to_unit: str,
    label: str = "",
) -> Optional[float]:
    """
    Attempts to convert scalar `value` from `from_unit` to `to_unit`.
    Returns converted float if conversion is known, or None if incompatible/unknown.
    """
    if pd.isna(value):
        return None

    u_from = canonicalize_unit(from_unit)
    u_to = canonicalize_unit(to_unit)

    if u_from == "" or u_to == "":
        return None
    if u_from == u_to:
        return float(value)

    if (u_from, u_to) in _GENERAL_CONVERSIONS:
        scale, offset = _GENERAL_CONVERSIONS[(u_from, u_to)]
        return float(value) * scale + offset

    lbl_lower = str(label).lower()
    for kw, unit_a, unit_b, factor in _ANALYTE_CONVERSIONS:
        if kw in lbl_lower:
            if u_from == unit_a and u_to == unit_b:
                return float(value) * factor
            if u_from == unit_b and u_to == unit_a:
                return float(value) / factor

            # 2-step conversion via general metric scaling (e.g., g/L -> g/dL -> mmol/L)
            if (u_from, unit_a) in _GENERAL_CONVERSIONS:
                s1, o1 = _GENERAL_CONVERSIONS[(u_from, unit_a)]
                return (float(value) * s1 + o1) * factor
            if (u_from, unit_b) in _GENERAL_CONVERSIONS:
                s1, o1 = _GENERAL_CONVERSIONS[(u_from, unit_b)]
                return (float(value) * s1 + o1) / factor
            if (unit_b, u_to) in _GENERAL_CONVERSIONS:
                s2, o2 = _GENERAL_CONVERSIONS[(unit_b, u_to)]
                return (float(value) * factor) * s2 + o2
            if (unit_a, u_to) in _GENERAL_CONVERSIONS:
                s2, o2 = _GENERAL_CONVERSIONS[(unit_a, u_to)]
                return (float(value) / factor) * s2 + o2

    # Monovalent electrolytes often share mEq/L <-> mmol/L 1:1 even without label match
    if {u_from, u_to} == {"mEq/L", "mmol/L"} and any(
        k in lbl_lower for k in ("sodium", "potassium", "chloride", "bicarbonate", "co2", "na", "k", "cl")
    ):
        return float(value)

    return None


def normalize_lab_units(
    labevents_df: pd.DataFrame,
    unit_registry: Dict[int, Dict[str, str]],
    emit_warnings: bool = True,
) -> Tuple[pd.DataFrame, UnitConversionReport]:
    """
    Normalizes and converts labevent units to match `unit_registry` (fitted on training data).

    - If a lab unit matches the training canonical unit (case/alias-insensitive), keeps as-is.
    - If a lab unit differs but is convertible, converts `valuenum` and updates `valueuom`.
    - If a lab unit differs and CANNOT be converted, emits a warning, records the mismatch,
      and sets `valuenum` to NaN so corrupted out-of-scale numbers do not enter quantile buckets.
    """
    report = UnitConversionReport()
    if labevents_df is None or labevents_df.empty:
        return labevents_df.copy() if labevents_df is not None else pd.DataFrame(), report

    df = labevents_df.copy()
    if "valueuom" not in df.columns:
        return df, report

    seen_conv_msgs = set()
    seen_warn_msgs = set()
    unseen_set = set()

    new_vals = df["valuenum"].tolist()
    new_uoms = df["valueuom"].tolist()
    itemids = df["itemid"].tolist()
    labels = df["label"].tolist() if "label" in df.columns else [""] * len(df)

    for idx, (iid, val, uom, row_lbl) in enumerate(zip(itemids, new_vals, new_uoms, labels)):
        if pd.isna(iid):
            continue
        iid_int = int(iid)
        if iid_int not in unit_registry:
            unseen_set.add(iid_int)
            continue

        expected_unit = unit_registry[iid_int].get("unit", "")
        if not expected_unit:
            continue

        actual_unit = canonicalize_unit(uom)
        if actual_unit == "":
            report.missing_unit_count += 1
            continue

        if actual_unit == expected_unit:
            new_uoms[idx] = expected_unit
            continue

        label_str = str(row_lbl) if pd.notnull(row_lbl) and str(row_lbl).strip() else unit_registry[iid_int].get("label", "")
        converted = convert_unit_value(val, actual_unit, expected_unit, label=label_str)

        if converted is not None:
            new_vals[idx] = converted
            new_uoms[idx] = expected_unit
            report.converted_count += 1
            conv_msg = f"itemid={iid_int} ({label_str or 'unknown'}): converted {actual_unit} -> {expected_unit}"
            if conv_msg not in seen_conv_msgs:
                seen_conv_msgs.add(conv_msg)
                report.conversions.append(conv_msg)
        else:
            # Unconvertible unit mismatch: do NOT feed unscaled numbers into quantile bucketizer
            new_vals[idx] = np.nan
            report.unconverted_count += 1
            warn_msg = (
                f"Unit mismatch for itemid={iid_int} ({label_str or 'unknown'}): "
                f"got '{actual_unit}', expected '{expected_unit}' (no valid conversion; value masked to NaN)."
            )
            if warn_msg not in seen_warn_msgs:
                seen_warn_msgs.add(warn_msg)
                report.warnings.append(warn_msg)
                if emit_warnings:
                    warnings.warn(warn_msg, UserWarning, stacklevel=2)

    df["valuenum"] = new_vals
    df["valueuom"] = new_uoms
    report.unseen_itemids = sorted(unseen_set)
    report.unseen_itemid_count = len(unseen_set)

    return df, report
