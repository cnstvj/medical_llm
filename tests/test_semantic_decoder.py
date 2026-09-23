"""
Unit tests for SemanticTransformerDecoder.
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
from medical_llm.semantic.decoder import SemanticTransformerDecoder


class TestSemanticDecoder(unittest.TestCase):

    def setUp(self):
        self.decoder = SemanticTransformerDecoder()

    def test_decode_high_risk_narrative(self):
        """Verify decoder produces accurate narrative from high-risk semantic representation."""
        rep = SemanticRepresentation(
            risk_level="high",
            mortality_probability=0.82,
            confidence_score=0.91,
            clinical_findings=[
                ClinicalFinding(
                    lab_name="Glucose",
                    direction="high",
                    statistical_bucket="Q4",
                    clinical_interpretation="elevated",
                    confidence=0.90,
                ),
                ClinicalFinding(
                    lab_name="Lactate",
                    direction="high",
                    statistical_bucket="Q4",
                    clinical_interpretation="elevated",
                    confidence=0.90,
                ),
            ],
            temporal_context=TemporalDynamics(
                interval_summary="acute intervals under 6 hours",
                dominant_gap="TIME_1TO6H",
            ),
            text_evidence=TextEvidence(
                trigger_keywords=["hypotension"]
            ),
            uncertainties=[
                UncertaintyStatement(
                    statement="The available information does not establish a specific diagnosis."
                )
            ],
        )

        res = self.decoder.decode_to_narrative(rep)
        narrative = res["full_explanation"]

        self.assertIn("high risk", narrative.lower())
        self.assertIn("82.0%", narrative)
        self.assertIn("glucose", narrative.lower())
        self.assertIn("lactate", narrative.lower())
        self.assertIn("does not establish a specific diagnosis", narrative.lower())

    def test_decode_low_risk_narrative(self):
        """Verify decoder articulates baseline stability for low-risk representation."""
        rep = SemanticRepresentation(
            risk_level="low",
            mortality_probability=0.10,
            confidence_score=0.89,
            clinical_findings=[],
            temporal_context=TemporalDynamics(interval_summary="observed admission timeframe"),
            uncertainties=[
                UncertaintyStatement(
                    statement="Statistical predictions do not constitute a clinical diagnosis."
                )
            ],
        )

        res = self.decoder.decode_to_narrative(rep)
        narrative = res["full_explanation"]

        self.assertIn("low risk", narrative.lower())
        self.assertIn("10.0%", narrative)
        self.assertIn("stability", narrative.lower())


if __name__ == "__main__":
    unittest.main()
