"""
Unit tests for BLEU (Bilingual Evaluation Understudy) metric implementation.
Covers modified n-gram precision, clipping, brevity penalty, smoothing,
multi-reference handling, and corpus-level aggregation.
"""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.utils.bleu import (
    sentence_bleu,
    sentence_bleu_detailed,
    corpus_bleu,
    corpus_bleu_detailed,
    compute_ngrams,
    modified_precision,
    calculate_brevity_penalty,
    closest_reference_length,
)
from medical_llm.semantic.metrics import SemanticEvaluationSuite
from medical_llm.semantic.representation import SemanticRepresentation, ClinicalFinding


class TestBLEUMetric(unittest.TestCase):

    def test_exact_match_score_is_one(self):
        """Exact hypothesis and reference must produce BLEU-4 of 1.0."""
        text = "The patient shows elevated creatinine and acute kidney injury risk."
        score = sentence_bleu(text, text, smoothing="none")
        self.assertAlmostEqual(score, 1.0, places=4)

        result = sentence_bleu_detailed(text, text, smoothing="none")
        self.assertAlmostEqual(result.score, 1.0, places=4)
        self.assertAlmostEqual(result.brevity_penalty, 1.0, places=4)
        for p in result.precisions:
            self.assertAlmostEqual(p, 1.0, places=4)

    def test_completely_disjoint_score_is_zero(self):
        """Candidate with zero matching words must produce BLEU of 0.0."""
        cand = "alpha beta gamma delta"
        ref = "one two three four"
        score = sentence_bleu(cand, ref, smoothing="none")
        self.assertEqual(score, 0.0)

    def test_modified_precision_clipping(self):
        """
        Modified precision must clip candidate repetitions to maximum reference count.
        Classic Papineni test case:
          Candidate: 'the the the the the the the' (7 words)
          Reference: 'the cat is on the mat' (contains 'the' twice)
          Unclipped precision: 7/7 = 1.0
          Clipped precision: 2/7
        """
        cand = ["the"] * 7
        refs = [["the", "cat", "is", "on", "the", "mat"]]

        clipped, total = modified_precision(cand, refs, n=1)
        self.assertEqual(clipped, 2)
        self.assertEqual(total, 7)

    def test_brevity_penalty_calculation(self):
        """Verify brevity penalty values for c > r, c == r, and c < r."""
        # When candidate is longer than reference: BP = 1.0
        self.assertAlmostEqual(calculate_brevity_penalty(10, 8), 1.0, places=5)
        # When candidate equals reference length: BP = 1.0
        self.assertAlmostEqual(calculate_brevity_penalty(8, 8), 1.0, places=5)
        # When candidate is shorter: BP = exp(1 - r / c)
        expected_bp = math.exp(1.0 - (10.0 / 5.0))  # exp(-1) = 0.367879...
        self.assertAlmostEqual(calculate_brevity_penalty(5, 10), expected_bp, places=5)
        # Empty candidate: BP = 0.0
        self.assertEqual(calculate_brevity_penalty(0, 5), 0.0)

    def test_closest_reference_length_selection(self):
        """Closest reference length selection and tie-breaking by shorter length."""
        # Candidate length 15, references [10, 22] -> closest is 10 (|10-15|=5 vs |22-15|=7)
        self.assertEqual(closest_reference_length(15, [10, 22]), 10)
        # Candidate length 16, references [12, 20] -> distances are both 4 -> tie-break chooses 12
        self.assertEqual(closest_reference_length(16, [12, 20]), 12)

    def test_multi_reference_scoring(self):
        """Candidate evaluation against multiple alternative references."""
        cand = "The patient has elevated serum lactate."
        refs = [
            "Patient exhibited high arterial lactate levels.",
            "The patient demonstrated elevated serum lactate.",
            "Lactate was significantly high in the patient.",
        ]
        # Candidate matches reference 2 closely
        res = sentence_bleu_detailed(cand, refs, smoothing="method1", lowercase=True)
        self.assertGreater(res.score, 0.45)
        self.assertAlmostEqual(res.brevity_penalty, 1.0, places=4)

        # Exact match to one reference should yield 1.0
        exact_score = sentence_bleu(
            "The patient demonstrated elevated serum lactate.",
            refs,
            smoothing="none",
            lowercase=True,
        )
        self.assertAlmostEqual(exact_score, 1.0, places=4)

    def test_smoothing_on_short_sentences(self):
        """
        Without smoothing, 4-gram precision on 3-word sentence is 0.0, zeroing standard BLEU.
        Smoothing method 1 replaces 0 with small epsilon, allowing non-zero score.
        """
        short_cand = "severe acute sepsis"
        ref = "severe acute sepsis developed during admission"

        strict_score = sentence_bleu(short_cand, ref, smoothing="none")
        self.assertEqual(strict_score, 0.0)  # No 4-grams exist in 3-word hypothesis

        smoothed_score = sentence_bleu(short_cand, ref, smoothing="method1")
        self.assertGreater(smoothed_score, 0.0)
        self.assertLessEqual(smoothed_score, 1.0)

    def test_bleu_n_gram_weights(self):
        """BLEU-1, BLEU-2, BLEU-3, BLEU-4 weight customization."""
        cand = "acute myocardial infarction with cardiogenic shock"
        ref = "acute myocardial infarction leading to cardiogenic shock"

        # BLEU-1 precision is 5/6 with BP = exp(1 - 7/6) = ~0.8465 -> score ~ 0.7054
        bleu_1 = sentence_bleu(cand, ref, weights=(1.0,), lowercase=True)
        self.assertAlmostEqual(bleu_1, 0.7054, places=3)

        # BLEU-2 evaluates unigrams and bigrams: sqrt(5/6 * 3/5) * BP ~= 0.5986
        bleu_2 = sentence_bleu(cand, ref, weights=(0.5, 0.5), lowercase=True)
        self.assertAlmostEqual(bleu_2, 0.5986, places=3)

    def test_corpus_bleu_aggregation(self):
        """
        Corpus BLEU sums clipped counts and lengths over all sentences before taking logarithm.
        """
        cands = [
            "The patient has elevated troponin.",
            "Kidney function shows elevated creatinine.",
        ]
        refs = [
            ["The patient has elevated troponin levels."],
            ["Kidney function shows high creatinine levels."],
        ]

        c_result = corpus_bleu_detailed(cands, refs, smoothing="method1", lowercase=True)
        self.assertGreater(c_result.score, 0.20)
        self.assertEqual(c_result.candidate_length, 12)
        self.assertGreater(c_result.reference_length, 12)
        self.assertLess(c_result.brevity_penalty, 1.0)

    def test_semantic_evaluation_suite_bleu_integration(self):
        """SemanticEvaluationSuite integrates compute_sentence_bleu and evaluate_batch with reference_texts."""
        cands = [
            "Elevated lactate and creatinine indicate severe hypoperfusion.",
            "Normal electrolyte panel with stable trajectory.",
        ]
        gold_refs = [
            "Elevated lactate and high creatinine indicate severe tissue hypoperfusion.",
            "Normal electrolyte panel and stable clinical trajectory.",
        ]

        # 1. Direct suite sentence call
        score_1 = SemanticEvaluationSuite.compute_sentence_bleu(cands[0], gold_refs[0])
        self.assertGreater(score_1, 0.20)

        # 2. Batch evaluation with references
        dummy_findings = [
            ClinicalFinding(
                lab_name="Lactate",
                direction="high",
                statistical_bucket="Q9",
                clinical_interpretation="Elevated lactate indicating tissue hypoperfusion",
            )
        ]
        dummy_rep = SemanticRepresentation(
            risk_level="high",
            mortality_probability=0.85,
            confidence_score=0.90,
            clinical_findings=dummy_findings,
            clinical_summary="High lactate",
        )
        batch_res = SemanticEvaluationSuite.evaluate_batch(
            explanations=cands,
            semantic_reps=[dummy_rep, dummy_rep],
            reference_texts=gold_refs,
        )

        self.assertIn("corpus_bleu_4", batch_res)
        self.assertIn("mean_sentence_bleu_4", batch_res)
        self.assertGreater(batch_res["corpus_bleu_4"], 0.15)
        self.assertGreater(batch_res["mean_sentence_bleu_4"], 0.15)


if __name__ == "__main__":
    unittest.main()

