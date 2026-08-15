"""
Preprocessing module for MIMIC-III admissions, patients, lab events, and clinical reports.
"""

import os
import pandas as pd
from medical_llm.config import DEFAULT_CONFIG, PipelineConfig
from medical_llm.data.text_utils import clean_text
from medical_llm.utils.quantiles import fit_quantiles, apply_quantile, save_quantiles


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


def process_labevents(
    labevents_df: pd.DataFrame,
    d_labitems_df: pd.DataFrame,
    quantiles_filename: str = DEFAULT_CONFIG.quantiles_file,
) -> pd.DataFrame:
    """Discretizes continuous lab measurement values into quantile buckets and generates event tokens."""
    labevents = labevents_df.dropna(subset=["hadm_id"])
    labevents = pd.merge(labevents, d_labitems_df[["itemid", "label"]], on="itemid", how="left")

    quantile_map = fit_quantiles(labevents)
    save_quantiles(quantile_map, filename=quantiles_filename)

    qbuckets = []
    for itemid, value in zip(labevents["itemid"], labevents["valuenum"]):
        if itemid in quantile_map:
            q = apply_quantile(value, quantile_map[itemid])
        else:
            q = None
        qbuckets.append(q)

    labevents["qbucket"] = qbuckets
    labevents["flag"] = labevents["flag"].fillna("NORMAL")

    tokens = []
    for q, f, item in zip(labevents["qbucket"], labevents["flag"], labevents["itemid"]):
        if pd.notnull(q):
            tokens.append(f"LAB_{item}_Q{int(q)}")
        else:
            tokens.append(f"LAB_{item}_{str(f).upper()}")

    labevents["event_token"] = tokens
    return labevents[["subject_id", "hadm_id", "charttime", "event_token", "valuenum"]]


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


def run_preprocessing(config: PipelineConfig = DEFAULT_CONFIG, dummy_mode: bool = False) -> None:
    """Main execution function to load raw CSVs, process all modal components, and output cleaned CSV files."""
    dummy_flag = dummy_mode or (os.environ.get("DUMMY_MODE", "") == "1")
    if dummy_flag:
        print("[DUMMY MODE] Using dummy CSV files.")

    adm_file = config.get_csv_name(config.admissions_csv, dummy_flag)
    pat_file = config.get_csv_name(config.patients_csv, dummy_flag)
    lab_file = config.get_csv_name(config.labevents_csv, dummy_flag)
    d_lab_file = config.get_csv_name(config.d_labitems_csv, dummy_flag)
    rep_file = config.get_csv_name(config.reports_csv, dummy_flag)

    print("Loading raw medical data CSVs...")
    admissions = pd.read_csv(adm_file)
    patients = pd.read_csv(pat_file)
    labevents = pd.read_csv(lab_file)
    d_labitems = pd.read_csv(d_lab_file)
    reports = pd.read_csv(rep_file)

    print("Merging admissions and patients...")
    admissions_processed = process_admissions_and_patients(admissions, patients)

    print("Processing lab events (Continuous Bucketing)...")
    labevents_processed = process_labevents(labevents, d_labitems, config.quantiles_file)

    print("Processing medical reports (NLP text normalization)...")
    reports_processed = process_clinical_reports(reports)

    print("Saving processed CSV artifacts...")
    admissions_processed.to_csv(config.processed_admissions_file, index=False)
    labevents_processed.to_csv(config.processed_labevents_file, index=False)
    reports_processed.to_csv(config.processed_reports_file, index=False)
    print("Preprocessing completed successfully.")
