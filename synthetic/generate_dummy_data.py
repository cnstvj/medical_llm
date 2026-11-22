"""
generate_dummy_data.py
======================
Generates small dummy CSV files that mirror the real MIMIC-III schema so the
full pipeline (preprocessing → sequence_builder → cross_validate) can be tested
quickly without the production dataset.

Files produced:
    ADMISSIONS_DUMMY.csv
    PATIENTS_DUMMY.csv
    LABEVENTS_DUMMY.csv
    D_LABITEMS_DUMMY.csv
    structured_medical_records_DUMMY.csv

Run:
    python generate_dummy_data.py
Then run the pipeline in test mode:
    python rerun_pipeline.py --dummy
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PATIENTS   = 40   # unique subjects
N_ADMISSIONS = 50   # admissions (some patients re-admitted)
N_LAB_ITEMS  = 20   # distinct lab item types
LABS_PER_ADM = 30   # avg lab events per admission
MORTALITY_RATE = 0.30  # fraction of admissions that end in death

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dummy")

ADMISSION_TYPES = ["EMERGENCY", "ELECTIVE", "URGENT"]
GENDERS         = ["M", "F"]
DIAGNOSES       = [
    "SEPSIS", "PNEUMONIA", "HEART FAILURE", "RENAL FAILURE",
    "LIVER DISEASE", "LUNG CANCER", "TRAUMA", "STROKE"
]
LAB_LABELS = [
    "Creatinine", "Sodium", "Potassium", "Bicarbonate", "Glucose",
    "Hemoglobin", "Hematocrit", "White Blood Cells", "Platelets",
    "Urea Nitrogen", "Calcium, Total", "Chloride", "Magnesium",
    "Phosphate", "Albumin", "Total Protein", "ALT", "AST",
    "Alkaline Phosphatase", "Bilirubin, Total"
]
FLUIDS     = ["Blood", "Urine", "CSF", "Other"]
CATEGORIES = ["Chemistry", "Hematology", "Blood Gas", "Microbiology"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rand_datetime(start: datetime, days: int) -> datetime:
    return start + timedelta(
        days=random.randint(0, days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )


def fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 1. PATIENTS
# ---------------------------------------------------------------------------
def make_patients(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        subject_id = 90000 + i
        gender     = random.choice(GENDERS)
        dob        = datetime(random.randint(1930, 1980),
                              random.randint(1, 12),
                              random.randint(1, 28))
        expire     = random.random() < 0.5
        dod        = rand_datetime(dob, 365 * 40) if expire else None
        rows.append({
            "row_id":      i + 1,
            "subject_id":  subject_id,
            "gender":      gender,
            "dob":         fmt(dob),
            "dod":         fmt(dod) if dod else "",
            "dod_hosp":    fmt(dod) if (dod and random.random() < 0.7) else "",
            "dod_ssn":     fmt(dod) if (dod and random.random() < 0.5) else "",
            "expire_flag": int(expire),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. ADMISSIONS
# ---------------------------------------------------------------------------
def make_admissions(patients: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    subject_ids = patients["subject_id"].tolist()
    used_hadm   = set()

    for i in range(n):
        hadm_id = 200000 + i
        while hadm_id in used_hadm:
            hadm_id += 1
        used_hadm.add(hadm_id)

        subject_id   = random.choice(subject_ids)
        admittime    = rand_datetime(datetime(2100, 1, 1), 365 * 5)
        los_days     = random.randint(1, 30)
        dischtime    = admittime + timedelta(days=los_days)
        expire_flag  = 1 if random.random() < MORTALITY_RATE else 0
        deathtime    = fmt(dischtime) if expire_flag else ""

        rows.append({
            "row_id":               i + 1,
            "subject_id":           subject_id,
            "hadm_id":              hadm_id,
            "admittime":            fmt(admittime),
            "dischtime":            fmt(dischtime),
            "deathtime":            deathtime,
            "admission_type":       random.choice(ADMISSION_TYPES),
            "admission_location":   "EMERGENCY ROOM ADMIT",
            "discharge_location":   "HOME" if not expire_flag else "DEAD/EXPIRED",
            "insurance":            random.choice(["Medicare", "Medicaid", "Private"]),
            "language":             "",
            "religion":             "",
            "marital_status":       random.choice(["MARRIED", "SINGLE", "WIDOWED"]),
            "ethnicity":            random.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC"]),
            "edregtime":            fmt(admittime - timedelta(hours=2)),
            "edouttime":            fmt(admittime),
            "diagnosis":            random.choice(DIAGNOSES),
            "hospital_expire_flag": expire_flag,
            "has_chartevents_data": 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. D_LABITEMS  (lab item dictionary)
# ---------------------------------------------------------------------------
def make_d_labitems() -> pd.DataFrame:
    rows = []
    for i, label in enumerate(LAB_LABELS):
        rows.append({
            "row_id":     i + 1,
            "itemid":     50800 + i,
            "label":      label,
            "fluid":      random.choice(FLUIDS),
            "category":   random.choice(CATEGORIES),
            "loinc_code": f"{10000 + i}-{i % 9}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. LABEVENTS
# ---------------------------------------------------------------------------
def make_labevents(admissions: pd.DataFrame,
                   d_labitems: pd.DataFrame) -> pd.DataFrame:
    rows = []
    row_id = 1
    itemids = d_labitems["itemid"].tolist()

    for _, adm in admissions.iterrows():
        admittime = pd.to_datetime(adm["admittime"])
        dischtime = pd.to_datetime(adm["dischtime"])
        n_labs = random.randint(LABS_PER_ADM // 2, LABS_PER_ADM * 2)
        span_secs = max(1, int((dischtime - admittime).total_seconds()))

        for _ in range(n_labs):
            charttime = admittime + timedelta(
                seconds=random.randint(0, span_secs)
            )
            itemid    = random.choice(itemids)
            valuenum  = round(random.uniform(0.5, 200.0), 2)
            flag      = random.choice(["abnormal", "", ""])  # mostly normal

            rows.append({
                "row_id":     row_id,
                "subject_id": adm["subject_id"],
                "hadm_id":    adm["hadm_id"],
                "itemid":     itemid,
                "charttime":  fmt(charttime),
                "value":      str(valuenum),
                "valuenum":   valuenum,
                "valueuom":   "mg/dL",
                "flag":       flag,
            })
            row_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. STRUCTURED MEDICAL REPORTS
# ---------------------------------------------------------------------------
def make_reports(admissions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, adm in admissions.iterrows():
        n_notes = random.randint(1, 3)
        for day in range(n_notes):
            note_time = pd.to_datetime(adm["admittime"]) + timedelta(days=day)
            report = (
                f"Patient ID: {adm['subject_id']}\r\n"
                f"    Admission Date: {pd.to_datetime(adm['admittime']).date()}\r\n"
                f"    Admission Type: {adm['admission_type']}\r\n"
                f"    Diagnosis: {adm['diagnosis']}\r\n"
                f"    **Lab Results:**\r\n"
                f"    - Glucose: {round(random.uniform(70, 250), 1)} mg/dL "
                f"({'abnormal' if random.random() > 0.5 else 'normal'})\r\n"
                f"    - Hemoglobin: {round(random.uniform(7, 16), 1)} g/dL "
                f"({'abnormal' if random.random() > 0.5 else 'normal'})\r\n"
                f"    - White Blood Cells: {round(random.uniform(3, 20), 1)} K/uL "
                f"({'abnormal' if random.random() > 0.5 else 'normal'})\r\n"
            )
            rows.append({
                "subject_id":     adm["subject_id"],
                "hadm_id":        adm["hadm_id"],
                "adm_date":       pd.to_datetime(adm["admittime"]).strftime("%Y/%m/%d"),
                "adm_time":       note_time.strftime("%H:%M:%S"),
                "medical_report": report,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generator API & CLI
# ---------------------------------------------------------------------------
def generate_dummy_datasets(out_dir: str = OUT_DIR, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Generates synthetic MIMIC-III compatible dummy tables and writes CSV files to out_dir."""
    if verbose:
        print("Generating dummy data...\n")

    patients   = make_patients(N_PATIENTS)
    admissions = make_admissions(patients, N_ADMISSIONS)
    d_labitems = make_d_labitems()
    labevents  = make_labevents(admissions, d_labitems)
    reports    = make_reports(admissions)

    out = {
        "admissions_dummy.csv":                 admissions,
        "patients_dummy.csv":                   patients,
        "d_labitems_dummy.csv":                 d_labitems,
        "labevents_dummy.csv":                  labevents,
        "structured_medical_records_dummy.csv": reports,
    }

    for fname, df in out.items():
        path = os.path.join(out_dir, fname)
        df.to_csv(path, index=False)
        if verbose:
            print(f"  Written {len(df):>5} rows  ->  {fname}")

    deaths = admissions["hospital_expire_flag"].sum()
    if verbose:
        print(f"\nDone.")
        print(f"  {N_ADMISSIONS} admissions")
        print(f"  {deaths} deaths ({100*deaths/N_ADMISSIONS:.1f}%)")
        print(f"  {len(labevents)} lab events")
        print(f"  {len(reports)} medical notes")

    return out


def main():
    generate_dummy_datasets(out_dir=OUT_DIR, verbose=True)


if __name__ == "__main__":
    main()

