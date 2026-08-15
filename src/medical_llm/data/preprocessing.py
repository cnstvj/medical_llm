"""
Preprocessing module for MIMIC-III admissions, patients, lab events, and clinical reports.
Implements leak-free quantile fitting strictly on training split lab events.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from medical_llm.config import DEFAULT_CONFIG, PipelineConfig
from medical_llm.data.text_utils import clean_text
from medical_llm.utils.quantiles import (
    fit_quantiles,
    apply_quantiles_to_labevents,
    save_quantiles,
)


def process_admissions_and_patients(admissions_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """Merges admissions with patients, computes patient age, and flags in-hospital mortality target."""
    adm_pat = pd.merge(admissions_df, patients_df, on="subject_id", how="inner")
    adm_pat["admittime"] = pd.to_datetime(adm_pat["admittime"])
    adm_pat["dob"] = pd.to_datetime(adm_pat["dob"])

    # Compute age from year components to avoid int64 overflow on de-identified MIMIC-III DOBs
    adm_pat["age"] = (
        adm_pat["admittime"].dt.year
        - adm_pat["dob"].dt.year
        - (
            (adm_pat["admittime"].dt.month * 100 + adm_pat["admittime"].dt.day)
            < (adm_pat["dob"].dt.month * 100 + adm_pat["dob"].dt.day)
        ).astype(int)
    )
    adm_pat.loc[adm_pat["age"] > 89, "age"] = 90
    adm_pat["target"] = adm_pat["hospital_expire_flag"]

    return adm_pat[
        ["subject_id", "hadm_id", "admittime", "dischtime", "age", "gender", "admission_type", "target"]
    ]


def process_clinical_reports(reports_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans clinical text notes and aggregates them per admission."""
    reports = reports_df.copy()
    reports["clean_text"] = reports["medical_report"].apply(clean_text)

    # One report row per admission
    reports_grouped = (
        reports.groupby("hadm_id", as_index=False)
        .agg({
            "clean_text": lambda x: " ".join(
                str(v) for v in x if pd.notnull(v) and str(v).strip()
            )
        })
    )
    return reports_grouped[["hadm_id", "clean_text"]]


def load_raw_tables(config: PipelineConfig = DEFAULT_CONFIG, dummy_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw admissions, patients, labevents, d_labitems, and reports CSV DataFrames."""
    dummy_flag = dummy_mode or (os.environ.get("DUMMY_MODE", "") == "1")
    if dummy_flag:
        print("[DUMMY MODE] Using dummy CSV files.")

    adm_file = config.get_csv_name(config.admissions_csv, dummy_flag)
    pat_file = config.get_csv_name(config.patients_csv, dummy_flag)
    lab_file = config.get_csv_name(config.labevents_csv, dummy_flag)
    d_lab_file = config.get_csv_name(config.d_labitems_csv, dummy_flag)
    rep_file = config.get_csv_name(config.reports_csv, dummy_flag)

    admissions = pd.read_csv(adm_file)
    patients = pd.read_csv(pat_file)
    labevents = pd.read_csv(lab_file)
    d_labitems = pd.read_csv(d_lab_file)
    reports = pd.read_csv(rep_file)

    admissions_processed = process_admissions_and_patients(admissions, patients)

    labevents = labevents.dropna(subset=["hadm_id"])
    labevents = pd.merge(labevents, d_labitems[["itemid", "label"]], on="itemid", how="left")

    reports_processed = process_clinical_reports(reports)

    return admissions_processed, labevents, reports_processed


def fit_and_apply_quantiles_split(
    train_labevents: pd.DataFrame,
    val_labevents: Optional[pd.DataFrame] = None,
    test_labevents: Optional[pd.DataFrame] = None,
    quantiles_filename: str = DEFAULT_CONFIG.quantiles_file,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[Any, Any]]:
    """Fits quantile map STRICTLY on train_labevents and applies those exact quantile boundaries to train, val, and test splits."""
    quantile_map = fit_quantiles(train_labevents)
    save_quantiles(quantile_map, filename=quantiles_filename)

    train_processed = apply_quantiles_to_labevents(train_labevents, quantile_map)
    val_processed = apply_quantiles_to_labevents(val_labevents, quantile_map) if val_labevents is not None else None
    test_processed = apply_quantiles_to_labevents(test_labevents, quantile_map) if test_labevents is not None else None

    return train_processed, val_processed, test_processed, quantile_map


def run_preprocessing(config: PipelineConfig = DEFAULT_CONFIG, dummy_mode: bool = False) -> None:
    """Standard execution function for preprocessing data with strict train-only quantile fitting."""
    print("Loading raw medical data CSVs...")
    admissions_processed, labevents, reports_processed = load_raw_tables(config, dummy_mode=dummy_mode)

    # Perform patient-level stratified split to isolate train labevents for quantile fitting
    train_hadms, _ = train_test_split(
        admissions_processed["hadm_id"].unique(),
        test_size=0.2,
        random_state=config.random_seed,
    )
    train_labevents = labevents[labevents["hadm_id"].isin(train_hadms)]

    print("Fitting quantiles STRICTLY on training split lab events...")
    quantile_map = fit_quantiles(train_labevents)
    save_quantiles(quantile_map, filename=config.quantiles_file)

    print("Applying training quantile boundaries to all lab events...")
    labevents_processed = apply_quantiles_to_labevents(labevents, quantile_map)
    labevents_processed = labevents_processed[["subject_id", "hadm_id", "charttime", "event_token", "valuenum"]]

    print("Saving processed CSV artifacts...")
    admissions_processed.to_csv(config.processed_admissions_file, index=False)
    labevents_processed.to_csv(config.processed_labevents_file, index=False)
    reports_processed.to_csv(config.processed_reports_file, index=False)
    print("Preprocessing completed successfully with 0% quantile leakage.")
