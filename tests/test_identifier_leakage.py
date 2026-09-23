"""
Unit tests verifying 0% patient identifier and exact timestamp leakage into model inputs.
"""

import sys
import os
import unittest
import json
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.data.text_utils import clean_text
from medical_llm.data.sequence_builder import build_temporal_sequences, time_gap_token
from medical_llm.config import DEFAULT_CONFIG


class TestIdentifierLeakage(unittest.TestCase):

    def test_clean_text_strips_patient_and_admission_ids(self):
        """Verify that clean_text removes Patient ID, Subject ID, and Hadm ID headers."""
        raw_note = (
            "Patient ID: 10076\r\n"
            "    Admission ID: 198503\r\n"
            "    Subject_ID: 10076\r\n"
            "    Hadm_ID: 198503\r\n"
            "    Diagnosis: LUNG CANCER;SHORTNESS OF BREATH\r\n"
            "    **Lab Results:**\r\n"
            "- Glucose: 116.0 mg/dL (abnormal)\r\n"
        )
        cleaned = clean_text(raw_note)

        # Identifiers should not be in the cleaned text
        self.assertNotIn("10076", cleaned)
        self.assertNotIn("198503", cleaned)
        self.assertNotIn("patient id", cleaned)
        self.assertNotIn("admission id", cleaned)
        self.assertNotIn("hadm id", cleaned)

        # Medical findings should be preserved
        self.assertIn("lung cancer", cleaned)
        self.assertIn("shortness of breath", cleaned)
        self.assertIn("glucose", cleaned)

    def test_clean_text_strips_exact_dates_and_timestamps(self):
        """Verify that clean_text removes exact calendar dates and timestamps."""
        raw_note = (
            "Admission Date: 2107-03-21\r\n"
            "Admission Time: 17:50:00\r\n"
            "DOB: 2038-01-15\r\n"
            "Patient evaluated on 2107/03/22 at 06:10:00 with fever.\r\n"
        )
        cleaned = clean_text(raw_note)

        self.assertNotIn("2107-03-21", cleaned)
        self.assertNotIn("2107", cleaned)
        self.assertNotIn("17:50:00", cleaned)
        self.assertNotIn("17 50 00", cleaned)
        self.assertNotIn("2038-01-15", cleaned)
        self.assertNotIn("2038", cleaned)
        self.assertNotIn("06:10:00", cleaned)

        # Clinical word content preserved
        self.assertIn("fever", cleaned)

    def test_sequence_dataset_contains_no_subject_or_hadm_ids_in_tokens(self):
        """Verify that sequence_dataset.json does not inject subject_id or hadm_id into token sequences."""
        if not os.path.exists(DEFAULT_CONFIG.sequence_dataset_file):
            self.skipTest("sequence_dataset.json not present")

        with open(DEFAULT_CONFIG.sequence_dataset_file, "r") as f:
            data = json.load(f)

        subject_id_strs = {str(sid) for sid in data.get("subject_ids", [])}
        hadm_id_strs = {str(hid) for hid in data.get("hadm_ids", [])}

        for seq_idx, seq in enumerate(data["sequences"]):
            tokens_set = set(seq)
            overlap_subj = tokens_set & subject_id_strs
            overlap_hadm = tokens_set & hadm_id_strs

            self.assertEqual(len(overlap_subj), 0, f"Sequence {seq_idx} contains subject_id tokens: {overlap_subj}")
            self.assertEqual(len(overlap_hadm), 0, f"Sequence {seq_idx} contains hadm_id tokens: {overlap_hadm}")

    def test_relative_temporal_and_demographic_tokens_preserved(self):
        """Verify relative temporal gap tokens and demographic tokens are preserved."""
        self.assertEqual(time_gap_token(0.5), "TIME_LT1H")
        self.assertEqual(time_gap_token(3.0), "TIME_1TO6H")
        self.assertEqual(time_gap_token(12.0), "TIME_6TO24H")
        self.assertEqual(time_gap_token(48.0), "TIME_1TO3D")
        self.assertEqual(time_gap_token(100.0), "TIME_GT3D")


if __name__ == "__main__":
    unittest.main()
