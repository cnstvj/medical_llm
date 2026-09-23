import sys
import os
import unittest
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
import pandas as pd
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.data.dataset import train_test_split_by_subject


class TestPatientLeakage(unittest.TestCase):

    def setUp(self):
        # Create synthetic dataset with repeat admissions per patient
        # Patients 1..10 have multiple admissions
        np.random.seed(42)
        records = []
        for sid in range(1, 41):  # 40 patients
            num_admissions = np.random.randint(1, 4)
            has_death = 1 if sid <= 12 else 0  # 12 patients have deaths
            for adm_idx in range(num_admissions):
                records.append({
                    "subject_id": sid,
                    "hadm_id": 1000 + len(records),
                    "target": has_death if adm_idx == num_admissions - 1 else 0,
                    "tokens": [f"TOKEN_{sid}_{adm_idx}"]
                })

        self.df = pd.DataFrame(records)
        self.seqs = self.df["tokens"].tolist()
        self.labels = self.df["target"].tolist()
        self.subject_ids = self.df["subject_id"].tolist()

    def test_train_test_split_zero_patient_overlap(self):
        """Verify train_test_split_by_subject produces 0 overlapping patients."""
        s_train, s_test, y_train, y_test, sid_train, sid_test = train_test_split_by_subject(
            self.seqs, self.labels, self.subject_ids, test_size=0.2, random_state=42
        )

        train_sids = set(sid_train)
        test_sids = set(sid_test)
        overlap = train_sids & test_sids

        self.assertEqual(len(overlap), 0, f"Overlapping patients found: {overlap}")
        self.assertEqual(len(train_sids) + len(test_sids), len(set(self.subject_ids)))

    def test_stratified_group_kfold_zero_patient_overlap(self):
        """Verify StratifiedGroupKFold produces 0 overlapping patients across all folds."""
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        groups = self.df["subject_id"].values
        labels = self.df["target"].values

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(self.df, labels, groups=groups), start=1):
            train_sids = set(self.df.iloc[train_idx]["subject_id"])
            test_sids = set(self.df.iloc[test_idx]["subject_id"])
            overlap = train_sids & test_sids

            self.assertEqual(len(overlap), 0, f"Fold {fold} had overlapping patients: {overlap}")


if __name__ == "__main__":
    unittest.main()
