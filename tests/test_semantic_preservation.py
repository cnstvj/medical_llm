"""
End-to-end integration tests for Semantic Preservation and Communication Integrity.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.models.clinical_explainer import ClinicalExplainer
from medical_llm.semantic.representation import SemanticRepresentation
from medical_llm.semantic.metrics import SemanticEvaluationSuite


class TestSemanticPreservation(unittest.TestCase):

    def test_end_to_end_preservation_high_risk(self):
        """Verify full pipeline preserves meaning and maintains 0% unsupported claims."""
        tokens = [
            "AGE_70", "GENDER_F", "ADM_EMERGENCY", "[SEP]",
            "LAB_50813_Q9", "TIME_LT1H", "LAB_50912_Q9", "TIME_1TO6H", "LAB_50809_Q8"
        ]
        mortality_prob = 0.82
        clinical_text = "Emergency arrival with hypotension and suspected infection."

        result = ClinicalExplainer.generate_patient_rationale(
            probability=mortality_prob,
            lab_tokens=tokens,
            clinical_text=clinical_text,
            validate=True,
        )

        # 1. Output structure assertions
        self.assertIn("semantic_representation", result)
        self.assertIn("clinical_narrative", result)
        self.assertIn("validation", result)

        val = result["validation"]
        self.assertTrue(val["is_valid"], f"Validation failed: {val.get('violations')}")
        self.assertEqual(val["hallucination_score"], 0.0)

        # 2. Semantic preservation assertions
        rep = SemanticRepresentation.from_dict(result["semantic_representation"])
        preservation = SemanticEvaluationSuite.compute_semantic_preservation(
            result["clinical_narrative"], rep
        )
        self.assertGreaterEqual(preservation, 0.90)

        # 3. Lab names properly decoded
        narrative = result["clinical_narrative"]
        self.assertIn("lactate", narrative.lower())
        self.assertIn("creatinine", narrative.lower())

        # 4. Uncertainty statement present
        self.assertIn("does not establish a specific diagnosis", narrative.lower())

    def test_batch_semantic_evaluation_suite(self):
        """Verify SemanticEvaluationSuite aggregates metrics accurately across multiple patients."""
        sample_patients = [
            {
                "tokens": ["LAB_50813_Q9", "TIME_LT1H", "LAB_50912_Q9"],
                "prob": 0.88,
                "text": "Hypotension noted.",
            },
            {
                "tokens": ["LAB_50983_Q1", "TIME_1TO6H", "LAB_50971_Q9"],
                "prob": 0.65,
                "text": "Electrolyte shifts.",
            },
            {
                "tokens": ["LAB_50813_Q4", "TIME_6TO24H", "LAB_50912_Q4"],
                "prob": 0.15,
                "text": "Stable trajectory.",
            },
        ]

        exps = []
        reps = []

        for p in sample_patients:
            res = ClinicalExplainer.generate_patient_rationale(
                probability=p["prob"],
                lab_tokens=p["tokens"],
                clinical_text=p["text"],
                validate=True,
            )
            reps.append(SemanticRepresentation.from_dict(res["semantic_representation"]))
            exps.append(res["clinical_narrative"])

        batch_metrics = SemanticEvaluationSuite.evaluate_batch(exps, reps)

        self.assertEqual(batch_metrics["num_evaluated"], 3)
        self.assertGreaterEqual(batch_metrics["semantic_preservation_score"], 0.85)
        self.assertEqual(batch_metrics["unsupported_claim_rate"], 0.0)
        self.assertEqual(batch_metrics["directional_accuracy"], 1.0)
        self.assertEqual(batch_metrics["validation_pass_rate"], 1.0)
        self.assertGreaterEqual(batch_metrics["clinical_faithfulness_index"], 0.90)


if __name__ == "__main__":
    unittest.main()
