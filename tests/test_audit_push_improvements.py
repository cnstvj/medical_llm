"""
Unit tests for the 6 push improvements:
1. Full Cohort 5-Fold StratifiedGroupKFold (Zero Patient Leakage across all 129 admissions)
2. Retaining and Encoding Admission Diagnosis in Preprocessing & Tabular Features
3. Structured Clinical Prompt Synthesis (Eliminating Note Presence Mortality Leakage)
4. Continuous Time2Vec Delta-Times Calculation & Multimodal Dataset Feeding
5. Leak-Free Training Set Optimal Threshold Optimization (Zero Test Snooping)
"""

import unittest
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from medical_llm.data.preprocessing import process_admissions_and_patients
from medical_llm.models.tabular_features import extract_tabular_features, DIAGNOSIS_INDICATORS
from medical_llm.pipeline.cross_validator import (
    synthesize_clinical_text,
    find_optimal_threshold,
    build_multimodal_fold_data,
    evaluate_multimodal_fold,
)
from medical_llm.data.sequence_builder import build_temporal_sequences
from medical_llm.data.dataset import MultiModalMedicalDataset
from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer


class TestAuditPushImprovements(unittest.TestCase):

    def test_diagnosis_retention_in_preprocessing(self):
        """Verify that process_admissions_and_patients retains diagnosis column."""
        adm = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [101, 102],
            "admittime": ["2150-01-01 08:00:00", "2150-02-01 10:00:00"],
            "dischtime": ["2150-01-05 18:00:00", "2150-02-03 14:00:00"],
            "deathtime": ["", ""],
            "admission_type": ["EMERGENCY", "ELECTIVE"],
            "diagnosis": ["SEPSIS", "CONGESTIVE HEART FAILURE"],
            "hospital_expire_flag": [0, 1],
        })
        pat = pd.DataFrame({
            "subject_id": [1, 2],
            "gender": ["M", "F"],
            "dob": ["2080-01-01 00:00:00", "2075-01-01 00:00:00"],
        })
        proc = process_admissions_and_patients(adm, pat)
        self.assertIn("diagnosis", proc.columns)
        self.assertEqual(proc.loc[proc["hadm_id"] == 101, "diagnosis"].iloc[0], "SEPSIS")
        self.assertEqual(proc.loc[proc["hadm_id"] == 102, "diagnosis"].iloc[0], "CONGESTIVE HEART FAILURE")

    def test_tabular_features_encode_diagnosis_indicators(self):
        """Verify that extract_tabular_features encodes Sepsis, Pneumonia, Heart Failure, etc."""
        adm = pd.DataFrame({
            "hadm_id": [201, 202, 203],
            "age": [65, 72, 50],
            "gender": ["M", "F", "M"],
            "admission_type": ["EMERGENCY", "EMERGENCY", "ELECTIVE"],
            "diagnosis": ["SEVERE SEPSIS; SEPTIC SHOCK", "PNEUMONIA; RESPIRATORY FAILURE", "CORONARY ARTERY DISEASE"],
            "target": [1, 1, 0],
        })
        labs = pd.DataFrame({
            "hadm_id": [201, 202, 203],
            "itemid": [50800, 50800, 50800],
            "valuenum": [12.0, 15.0, 10.0],
        })
        X, y, feature_names = extract_tabular_features(adm, labs)

        for diag_key in DIAGNOSIS_INDICATORS.keys():
            self.assertIn(diag_key, feature_names)

        sepsis_idx = feature_names.index("diag_sepsis")
        pneumonia_idx = feature_names.index("diag_pneumonia")
        resp_idx = feature_names.index("diag_respiratory")

        # Patient 201 has Sepsis
        self.assertEqual(X[0, sepsis_idx], 1.0)
        self.assertEqual(X[0, pneumonia_idx], 0.0)

        # Patient 202 has Pneumonia and Respiratory
        self.assertEqual(X[1, sepsis_idx], 0.0)
        self.assertEqual(X[1, pneumonia_idx], 1.0)
        self.assertEqual(X[1, resp_idx], 1.0)

        # Patient 203 has neither
        self.assertEqual(X[2, sepsis_idx], 0.0)
        self.assertEqual(X[2, pneumonia_idx], 0.0)

    def test_synthesize_clinical_text_empty_and_non_empty(self):
        """Verify prompt synthesis provides semantic clinical text across 100% of admissions."""
        # Case 1: Empty note -> structured synthesis
        txt_empty = synthesize_clinical_text(
            age=68, gender="M", admission_type="EMERGENCY", diagnosis="SEPSIS", raw_text=""
        )
        self.assertEqual(txt_empty, "68-year-old male admitted via emergency with SEPSIS.")

        # Case 2: Non-empty note -> prepend diagnosis
        txt_with_note = synthesize_clinical_text(
            age=75, gender="F", admission_type="ELECTIVE", diagnosis="PNEUMONIA", raw_text="Patient presents with cough."
        )
        self.assertEqual(txt_with_note, "Diagnosis: PNEUMONIA. Patient presents with cough.")

    def test_continuous_delta_times_calculation_and_dataset_tensor(self):
        """Verify delta_times calculation between consecutive labs and MultiModalMedicalDataset tensor shape."""
        charttimes = pd.to_datetime(["2150-01-01 08:00:00", "2150-01-01 12:00:00", "2150-01-02 00:00:00"])
        synthetic_labs = pd.DataFrame({
            "hadm_id": [301, 301, 301],
            "charttime": charttimes,
            "event_token": ["LAB_50800_Q1", "LAB_50800_Q2", "LAB_50800_Q3"],
        })
        grouped = build_temporal_sequences(synthetic_labs, experiment_mode="FULL")
        self.assertIn("delta_times", grouped.columns)
        deltas = grouped.iloc[0]["delta_times"]

        # Tokens: LAB_1, TIME_1TO6H (4h), LAB_2, TIME_6TO24H (12h), LAB_3
        # Deltas: 0.0, 4.0, 4.0, 12.0, 12.0
        self.assertEqual(len(deltas), len(grouped.iloc[0]["event_token"]))
        self.assertEqual(deltas[0], 0.0)
        self.assertEqual(deltas[1], 4.0)
        self.assertEqual(deltas[3], 12.0)

        # Test MultiModalMedicalDataset with delta_times
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "AGE_60": 4, "GENDER_M": 5, "ADM_EMERGENCY": 6}
        for tok in grouped.iloc[0]["event_token"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        seq = ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]"] + grouped.iloc[0]["event_token"]
        seq_deltas = [0.0, 0.0, 0.0, 0.0] + deltas

        ds = MultiModalMedicalDataset(
            [seq], ["Sample note"], [1], vocab=vocab, max_lab_len=16, delta_times=[seq_deltas]
        )
        sample = ds[0]
        self.assertIn("delta_times", sample)
        self.assertEqual(sample["delta_times"].shape, (16,))
        self.assertEqual(sample["lab_ids"].shape, (16,))
        self.assertFalse(torch.isnan(sample["delta_times"]).any())

    def test_find_optimal_threshold_zero_snooping(self):
        """Verify optimal threshold search works strictly on training predictions."""
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs_train = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        th = find_optimal_threshold(y_train, probs_train, metric="f1")
        # Optimal threshold separating 0s and 1s should be between 0.4 and 0.6 (e.g. 0.45 - 0.60)
        self.assertTrue(0.4 <= th <= 0.6)

        # Apply strictly out-of-sample to test set
        y_test = np.array([0, 1])
        probs_test = np.array([0.3, 0.7])
        preds_test = (probs_test >= th).astype(int)
        self.assertEqual(list(preds_test), [0, 1])

    def test_full_cohort_stratified_group_kfold_isolation(self):
        """Verify StratifiedGroupKFold on real data admissions achieves 0 patient leakage across 5 folds."""
        admissions = pd.read_csv("data/real/admissions.csv")
        self.assertEqual(len(admissions), 129)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        groups = admissions["subject_id"].values
        labels = admissions["hospital_expire_flag"].values

        total_test_adm = 0
        all_test_sids = []

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(admissions, labels, groups=groups), start=1):
            train_adm = admissions.iloc[train_idx]
            test_adm = admissions.iloc[test_idx]

            train_sids = set(train_adm["subject_id"])
            test_sids = set(test_adm["subject_id"])
            overlap = train_sids & test_sids
            self.assertEqual(len(overlap), 0, f"Fold {fold} has overlapping patients: {overlap}")

            total_test_adm += len(test_adm)
            all_test_sids.extend(list(test_sids))

        # Reclaimed full cohort: sum of test admissions across all 5 folds must be exactly 129
        self.assertEqual(total_test_adm, 129)
        # All patients evaluated exactly once across 5 folds
        self.assertEqual(len(all_test_sids), len(set(groups)))


if __name__ == "__main__":
    unittest.main()
