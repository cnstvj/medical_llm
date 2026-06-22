import pandas as pd
import json

# =====================================================
# EXPERIMENT MODES
# =====================================================
#
# "DEMOGRAPHICS"
# "LABS"
# "TEMPORAL"
# "NOTES"
# "FULL"
#
# =====================================================

EXPERIMENT_MODE = "FULL"


def time_gap_token(delta_hours):
    if delta_hours < 1: return "TIME_LT1H"
    elif delta_hours < 6: return "TIME_1TO6H"
    elif delta_hours < 24: return "TIME_6TO24H"
    elif delta_hours < 72: return "TIME_1TO3D"
    return "TIME_GT3D"


def build_temporal_sequences(labevents):
    temporal_sequences = []

    for hadm_id, group in labevents.groupby("hadm_id"):
        group = group.sort_values("charttime")
        tokens = []
        previous_time = None

        for _, row in group.iterrows():
            current_time = row["charttime"]

            if EXPERIMENT_MODE == "LABS":
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


def main():
    print("=" * 60)
    print(f"BUILDING SEQUENCES ({EXPERIMENT_MODE})")
    print("=" * 60)

    admissions = pd.read_csv("processed_admissions.csv")
    labevents = pd.read_csv("processed_labevents.csv")
    reports = pd.read_csv("processed_reports.csv")

    labevents["charttime"] = pd.to_datetime(labevents["charttime"])
    labevents = labevents.sort_values(["hadm_id", "charttime"])

    grouped = build_temporal_sequences(labevents)

    df = pd.merge(admissions, grouped, on="hadm_id", how="left")
    df["event_token"] = df["event_token"].apply(lambda x: x if isinstance(x, list) else [])

    df = pd.merge(df, reports, on="hadm_id", how="left")

    sequences = []
    labels = []
    subject_ids = []
    hadm_ids = []

    for _, row in df.iterrows():

        hadm_id = int(row["hadm_id"])
        subject_id = int(row["subject_id"]) if pd.notnull(row["subject_id"]) else -1

        age_bucket = int(row["age"] // 10) * 10 if pd.notnull(row["age"]) else 0

        age_token = f"AGE_{age_bucket}"
        gender_token = f"GENDER_{row['gender']}"
        adm_type_token = f"ADM_{str(row['admission_type']).replace(' ', '_')}"

        seq = []

        # =================================================
        # DEMOGRAPHICS
        # =================================================

        if EXPERIMENT_MODE in ("DEMOGRAPHICS", "FULL"):
            seq.extend([age_token, gender_token, adm_type_token])

        # =================================================
        # LABS
        # =================================================

        if EXPERIMENT_MODE in ("LABS", "TEMPORAL", "FULL"):
            seq.append("[SEP]")
            seq.extend(row["event_token"])

        # =================================================
        # NOTES
        # =================================================

        if EXPERIMENT_MODE in ("NOTES", "FULL") and pd.notnull(row["clean_text"]) and row["clean_text"] != "":
            seq.append("[REPORT_START]")
            seq.extend(row["clean_text"].split())

        label = int(row["target"]) if pd.notnull(row["target"]) else 0

        sequences.append(seq)
        labels.append(label)
        subject_ids.append(subject_id)
        hadm_ids.append(hadm_id)

    dataset = {
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "sequences": sequences,
        "labels": labels
    }

    with open("sequence_dataset.json", "w") as f:
        json.dump(dataset, f)

    print(f"Generated {len(sequences)} sequences")
    print(f"Unique Patients : {len(set(subject_ids))}")
    print(f"Admissions      : {len(set(hadm_ids))}")
    print(f"Deaths          : {sum(labels)}")
    print(f"Alive           : {len(labels) - sum(labels)}")
    print("Saved sequence_dataset.json")


if __name__ == "__main__":
    main()