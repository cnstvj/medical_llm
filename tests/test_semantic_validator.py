"""
Unit tests for ExplanationValidator and medical claim checking.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.semantic.representation import (
    SemanticRepresentation,
    ClinicalFinding,
    TemporalDynamics,
    TextEvidence,
    UncertaintyStatement,
)
from medical_llm.semantic.validator import ExplanationValidator


class TestSemanticValidator(unittest.TestCase):

    def setUp(self):
        self.rep = SemanticRepresentation(
            risk_level="high",
            mortality_probability=0.85,
            confidence_score=0.92,
            clinical_findings=[
                ClinicalFinding(
                    lab_name="Glucose",
                    direction="high",
                    statistical_bucket="Q9",
                    clinical_interpretation="elevated glucose",
                ),
                ClinicalFinding(
                    lab_name="Lactate",
                    direction="high",
                    statistical_bucket="Q9",
                    clinical_interpretation="elevated lactate",
                ),
            ],
            temporal_context=TemporalDynamics(interval_summary="acute intervals under 6 hours"),
            text_evidence=TextEvidence(trigger_keywords=["hypotension"]),
            uncertainties=[
                UncertaintyStatement(
                    statement="These findings do not establish a specific diagnosis and require bedside clinical correlation."
                )
            ],
        )

    def test_valid_explanation_passes(self):
        """Verify faithful explanation passes validation without violations."""
        valid_text = (
            "The model estimates a high risk of in-hospital mortality (85.0%). "
            "Main laboratory findings include elevated glucose and elevated lactate. "
            "These findings occurred within acute intervals under 6 hours. "
            "Clinical notes noted hypotension. "
            "These deviations do not establish a specific diagnosis and require clinical correlation."
        )
        res = ExplanationValidator.validate(valid_text, self.rep)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.violations), 0)
        self.assertEqual(res.hallucination_score, 0.0)

    def test_unsupported_diagnosis_flagged(self):
        """Verify hallucinated diagnosis (e.g. diabetes) is detected and rejected."""
        invalid_text = (
            "The model estimates high risk. The patient has diabetes. "
            "Glucose is elevated. This does not establish a specific diagnosis."
        )
        res = ExplanationValidator.validate(invalid_text, self.rep)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("UNSUPPORTED_DIAGNOSIS" in v and "diabetes" in v for v in res.violations))

    def test_unsupported_treatment_action_flagged(self):
        """Verify unauthorized clinical treatment recommendation is flagged."""
        invalid_text = (
            "The model estimates high risk. Administer insulin immediately to treat the patient. "
            "Glucose is elevated. Findings do not establish a specific diagnosis."
        )
        res = ExplanationValidator.validate(invalid_text, self.rep)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("UNSUPPORTED_TREATMENT" in v for v in res.violations))

    def test_directional_contradiction_flagged(self):
        """Verify inverted lab direction (calling high lactate 'reduced lactate') is flagged."""
        invalid_text = (
            "The model estimates high risk with reduced lactate. "
            "These findings do not establish a specific diagnosis."
        )
        res = ExplanationValidator.validate(invalid_text, self.rep)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("DIRECTIONAL_CONTRADICTION" in v for v in res.violations))

    def test_missing_uncertainty_flagged(self):
        """Verify explanation lacking uncertainty boundary is flagged."""
        invalid_text = (
            "The model estimates a high risk of in-hospital mortality. "
            "Glucose and lactate are elevated."
        )
        res = ExplanationValidator.validate(invalid_text, self.rep)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("MISSING_UNCERTAINTY_CAVEAT" in v for v in res.violations))


if __name__ == "__main__":
    unittest.main()
