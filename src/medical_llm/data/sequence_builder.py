"""
Sequence builder module for creating temporal token sequences from processed MIMIC-III data.
"""

import json
from typing import Dict, Any, List
import pandas as pd
from medical_llm.config import DEFAULT_CONFIG, PipelineConfig


def time_gap_token(delta_hours: float) -> str:
    """Categorizes time difference into discrete temporal gap tokens."""
    if delta_hours < 1:
        return "TIME_LT1H"
    elif delta_hours < 6:
        return "TIME_1TO6H"
    elif delta_hours < 24:
        return "TIME_6TO24H"
    elif delta_hours < 72:
        return "TIME_1TO3D"
    else:
        return "TIME_GT3D"


def build_temporal_sequences(labevents: pd.DataFrame, experiment_mode: str = "FULL") -> pd.DataFrame:
    """Groups lab events by admission ID, sorts by charttime, and creates sequence of event and time tokens."""
    temporal_sequences = []

    for hadm_id, group in labevents.groupby("hadm_id"):
        group = group.sort_values("charttime")
        tokens = []
        previous_time = None

        for _, row in group.iterrows():
            current_time = row["charttime"]

            if experiment_mode == "LABS":
                tokens.append(row["event_token"])
            else:
                if previous_time is not None:
                    delta_hours = (current_time - previous_time).total_seconds() / 3600
                    tokens.append(time_gap_token(delta_hours))
                tokens.append(row["event_token"])

            previous_time = current_time

        temporal_sequences.append({
            "hadm_id": hadm_id,
            "event_token": tokens
        })

    return pd.DataFrame(temporal_sequences)


def run_sequence_builder(
    config: PipelineConfig = DEFAULT_CONFIG,
    experiment_mode: str = None
) -> Dict[str, Any]:
    """Builds token sequences and target labels according to experiment mode and saves JSON output."""
    mode = experiment_mode or config.experiment_mode
    print("=" * 50)
    print(f"Building sequences ({mode})")
    print("=" * 50)

    admissions = pd.read_csv(config.processed_admissions_file)
    labevents = pd.read_csv(config.processed_labevents_file)
    reports = pd.read_csv(config.processed_reports_file)

    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    labevents = labevents.sort_values(by=["hadm_id", "charttime"])

    grouped = build_temporal_sequences(labevents, experiment_mode=mode)

    df = pd.merge(admissions, grouped, on="hadm_id", how="left")
    df["event_token"] = df["event_token"].apply(lambda x: x if isinstance(x, list) else [])
    df = pd.merge(df, reports, on="hadm_id", how="left")

    sequences = []
    labels = []
    subject_ids = []
    hadm_ids = []

    for _, row in df.iterrows():
        age_bucket = int(row["age"] // 10) * 10 if pd.notnull(row["age"]) else 0
        age_token = f"AGE_{age_bucket}"
        gender_token = f"GENDER_{row['gender']}"
        adm_type_token = f"ADM_{str(row['admission_type']).replace(' ', '_')}"

        seq = []

        # DEMOGRAPHICS
        if mode in ("DEMOGRAPHICS", "FULL"):
            seq.extend([age_token, gender_token, adm_type_token])

        # LABS / TEMPORAL
        if mode in ("LABS", "TEMPORAL", "FULL"):
            seq.append("[SEP]")
            seq.extend(row["event_token"])

        # NOTES
        if mode in ("NOTES", "FULL"):
            if pd.notnull(row["clean_text"]) and str(row["clean_text"]).strip() != "":
                seq.append("[REPORT_START]")
                seq.extend(row["clean_text"].split())

        sequences.append(seq)
        labels.append(int(row["target"]) if pd.notnull(row["target"]) else 0)
        subject_ids.append(int(row["subject_id"]))
        hadm_ids.append(int(row["hadm_id"]))

    dataset = {
        "sequences": sequences,
        "labels": labels,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
    }

    with open(config.sequence_dataset_file, "w") as f:
        json.dump(dataset, f)

    print(f"Generated {len(sequences)} sequences into {config.sequence_dataset_file}")
    print(f"  Unique subjects: {len(set(subject_ids))}, Unique admissions: {len(set(hadm_ids))}")
    return dataset
