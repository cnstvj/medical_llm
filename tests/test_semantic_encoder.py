"""
Unit tests for SemanticTransformerEncoder and ClinicalDictionary token decoding.
"""

import sys
import os
import unittest

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.semantic.dictionary import ClinicalDictionary, get_default_dictionary
from medical_llm.semantic.encoder import SemanticTransformerEncoder
from medical_llm.semantic.representation import SemanticRepresentation


class TestSemanticEncoder(unittest.TestCase):

    def setUp(self):
        self.dictionary = get_default_dictionary()
        self.encoder = SemanticTransformerEncoder(dictionary=self.dictionary)

    def test_decode_lab_token_mapping(self):
        """Verify lab tokens are mapped to clinical names, fluids, and categories."""
        # Glucose / specimen
        dec_50800 = self.dictionary.decode_lab_token("LAB_50800_Q4")
        self.assertIsNotNone(dec_50800)
        self.assertEqual(dec_50800["itemid"], 50800)
        self.assertIn("SPECIMEN TYPE", dec_50800["lab_name"].upper())

        # Lactate
        dec_50813 = self.dictionary.decode_lab_token("LAB_50813_Q9")
        self.assertIsNotNone(dec_50813)
        self.assertEqual(dec_50813["itemid"], 50813)
        self.assertEqual(dec_50813["lab_name"], "Lactate")
        self.assertEqual(dec_50813["direction"], "high")
        self.assertEqual(dec_50813["significance"], "acute_high")

        # Creatinine
        dec_50912 = self.dictionary.decode_lab_token("LAB_50912_Q9")
        self.assertIsNotNone(dec_50912)
        self.assertEqual(dec_50912["itemid"], 50912)
        self.assertEqual(dec_50912["lab_name"], "Creatinine")
        self.assertEqual(dec_50912["direction"], "high")

    def test_statistical_quantile_vs_clinical_abnormality(self):
        """Verify deciles are distinguished from absolute clinical reference standards."""
        dec_q9 = self.dictionary.decode_lab_token("LAB_50813_Q9")
        self.assertTrue(dec_q9["is_statistical_decile"])
        self.assertIn("cohort distribution", dec_q9["distinction_note"].lower())

        dec_normal = self.dictionary.decode_lab_token("LAB_50813_NORMAL")
        self.assertFalse(dec_normal["is_statistical_decile"])

    def test_encode_patient_high_risk(self):
        """Verify encoder constructs complete structured representation for high-risk patient."""
        tokens = [
            "AGE_70", "GENDER_F", "ADM_EMERGENCY", "[SEP]",
            "LAB_50813_Q9", "TIME_LT1H", "LAB_50912_Q9", "TIME_1TO6H", "LAB_50809_Q8"
        ]
        prob = 0.85
        clinical_text = "Patient presented in septic shock with hypotension."

        rep = self.encoder.encode_patient(
            tokens=tokens,
            mortality_prob=prob,
            clinical_text=clinical_text,
        )

        self.assertIsInstance(rep, SemanticRepresentation)
        self.assertEqual(rep.risk_level, "high")
        self.assertEqual(rep.mortality_probability, 0.85)
        self.assertGreater(len(rep.clinical_findings), 0)

        # Check finding properties
        finding_names = [f.lab_name for f in rep.clinical_findings]
        self.assertIn("Lactate", finding_names)
        self.assertIn("Creatinine", finding_names)

        # Check temporal dynamics
        self.assertIn("acute", rep.temporal_context.trajectory_pacing)

        # Check text evidence
        self.assertIn("sepsis", rep.text_evidence.trigger_keywords)
        self.assertIn("hypotension", rep.text_evidence.trigger_keywords)

        # Check mandatory uncertainty
        self.assertTrue(any("does not establish a specific diagnosis" in u.statement for u in rep.uncertainties))

    def test_encode_patient_low_risk(self):
        """Verify encoder constructs appropriate representation for low-risk patient."""
        tokens = [
            "AGE_30", "GENDER_M", "ADM_ELECTIVE", "[SEP]",
            "LAB_50813_Q3", "TIME_6TO24H", "LAB_50912_Q4", "TIME_1TO3D"
        ]
        rep = self.encoder.encode_patient(tokens=tokens, mortality_prob=0.12)
        self.assertEqual(rep.risk_level, "low")
        self.assertIn("stable", rep.clinical_summary.lower())


if __name__ == "__main__":
    unittest.main()
