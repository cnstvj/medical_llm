import sys
import os
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.data.preprocessing import process_admissions_and_patients, load_raw_tables
from medical_llm.data.sequence_builder import time_gap_token, build_temporal_sequences
from medical_llm.data.dataset import MedicalDataset
from medical_llm.config import DEFAULT_CONFIG


class TestTemporalLeakage(unittest.TestCase):

    def setUp(self):
        # Create synthetic admissions: 1 alive, 1 deceased
        self.admissions = pd.DataFrame({
            "subject_id": [101, 102],
            "hadm_id": [201, 202],
            "admittime": ["2150-01-01 08:00:00", "2150-02-01 10:00:00"],
            "dischtime": ["2150-01-05 18:00:00", "2150-02-03 14:00:00"],
            "deathtime": ["", "2150-02-03 12:00:00"],  # Died at 12:00, discharged/coded at 14:00
            "admission_type": ["EMERGENCY", "EMERGENCY"],
            "hospital_expire_flag": [0, 1],
        })

        self.patients = pd.DataFrame({
            "subject_id": [101, 102],
            "gender": ["M", "F"],
            "dob": ["2080-01-01 00:00:00", "2075-01-01 00:00:00"],
            "expire_flag": [0, 1],
        })

        # Create synthetic lab events:
        # Adm 201:
        #   - 2150-01-01 06:00:00 (2h before admit -> valid ED lab)
        #   - 2150-01-03 10:00:00 (during stay -> valid)
        #   - 2150-01-06 10:00:00 (post-discharge -> INVALID)
        # Adm 202:
        #   - 2150-02-02 08:00:00 (during stay -> valid)
        #   - 2150-02-03 13:00:00 (post-death at 12:00 -> INVALID)
        self.labs = pd.DataFrame({
            "row_id": range(1, 6),
            "subject_id": [101, 101, 101, 102, 102],
            "hadm_id": [201, 201, 201, 202, 202],
            "itemid": [50800, 50801, 50802, 50800, 50801],
            "charttime": [
                "2150-01-01 06:00:00",
                "2150-01-03 10:00:00",
                "2150-01-06 10:00:00",
                "2150-02-02 08:00:00",
                "2150-02-03 13:00:00",
            ],
            "valuenum": [100.0, 110.0, 120.0, 95.0, 99.0],
            "flag": ["NORMAL"] * 5,
        })

        self.d_labitems = pd.DataFrame({
            "itemid": [50800, 50801, 50802],
            "label": ["Lab A", "Lab B", "Lab C"],
        })

    def test_cutoff_time_calculation(self):
        """Verify that cutoff_time is min(dischtime, deathtime) for deceased patients."""
        proc_adm = process_admissions_and_patients(self.admissions, self.patients)

        # Patient 101 (alive): cutoff is dischtime
        adm_101 = proc_adm[proc_adm["hadm_id"] == 201].iloc[0]
        self.assertEqual(adm_101["cutoff_time"], pd.to_datetime("2150-01-05 18:00:00"))

        # Patient 102 (deceased at 12:00 vs disch at 14:00): cutoff is deathtime 12:00
        adm_102 = proc_adm[proc_adm["hadm_id"] == 202].iloc[0]
        self.assertEqual(adm_102["cutoff_time"], pd.to_datetime("2150-02-03 12:00:00"))

    def test_post_discharge_and_post_death_labs_filtered(self):
        """Verify that post-discharge and post-death labs are completely filtered out."""
        proc_adm = process_admissions_and_patients(self.admissions, self.patients)

        labs = self.labs.copy()
        labs["charttime"] = pd.to_datetime(labs["charttime"])
        merged = pd.merge(labs, proc_adm[["hadm_id", "admittime", "cutoff_time"]], on="hadm_id", how="inner")

        valid_mask = (
            (merged["charttime"] <= merged["cutoff_time"]) &
            (merged["charttime"] >= merged["admittime"] - pd.Timedelta(hours=24))
        )
        filtered_labs = merged[valid_mask]

        # Should retain only 3 valid labs (201: 2 labs, 202: 1 lab)
        self.assertEqual(len(filtered_labs), 3)
        self.assertNotIn(pd.to_datetime("2150-01-06 10:00:00"), filtered_labs["charttime"].values)
        self.assertNotIn(pd.to_datetime("2150-02-03 13:00:00"), filtered_labs["charttime"].values)

    def test_time_gap_token_bins(self):
        """Verify discrete temporal gap tokens mapping."""
        self.assertEqual(time_gap_token(0.5), "TIME_LT1H")
        self.assertEqual(time_gap_token(3.0), "TIME_1TO6H")
        self.assertEqual(time_gap_token(12.0), "TIME_6TO24H")
        self.assertEqual(time_gap_token(48.0), "TIME_1TO3D")
        self.assertEqual(time_gap_token(100.0), "TIME_GT3D")

    def test_sequence_truncation_preserves_admission_demographics(self):
        """Verify MedicalDataset truncates from the tail, preserving [CLS] and admission demographics."""
        vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "AGE_70": 4, "GENDER_M": 5, "ADM_EMERGENCY": 6}
        for i in range(1000):
            vocab[f"TOKEN_{i}"] = 7 + i

        # Create a long sequence with demographics at the start
        long_seq = ["AGE_70", "GENDER_M", "ADM_EMERGENCY", "[SEP]"] + [f"TOKEN_{i}" for i in range(600)]
        dataset = MedicalDataset([long_seq], [1], vocab, max_len=512)

        x, mask, y = dataset[0]

        # x[0] should be [CLS] (id 1)
        self.assertEqual(x[0].item(), 1)
        # x[1], x[2], x[3] should be AGE_70, GENDER_M, ADM_EMERGENCY
        self.assertEqual(x[1].item(), vocab["AGE_70"])
        self.assertEqual(x[2].item(), vocab["GENDER_M"])
        self.assertEqual(x[3].item(), vocab["ADM_EMERGENCY"])
        # Total tensor length should be max_len = 512
        self.assertEqual(len(x), 512)


if __name__ == "__main__":
    unittest.main()
