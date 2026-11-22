"""
Evaluation Metrics Suite for Semantic Communication and Clinical Explanation Quality.
Measures semantic preservation, unsupported-claim rate, directional accuracy,
uncertainty coverage, and composite clinical faithfulness.
"""

from typing import List, Dict, Any, Tuple, Optional, Sequence, Union
import re
import numpy as np

from medical_llm.semantic.representation import SemanticRepresentation
from medical_llm.semantic.validator import ExplanationValidator, ValidationResult
from medical_llm.utils.bleu import (
    sentence_bleu,
    corpus_bleu,
    sentence_bleu_detailed,
    corpus_bleu_detailed,
    BLEUResult,
)


class SemanticEvaluationSuite:
    """Evaluates semantic communication fidelity between technical representations and natural language explanations."""

    @staticmethod
    def compute_sentence_bleu(
        explanation: str,
        references: Union[str, Sequence[str]],
        weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        smoothing: Optional[str] = "method1",
        lowercase: bool = True,
    ) -> float:
        """Computes sentence-level BLEU-4 score between a generated explanation and reference text(s)."""
        return sentence_bleu(
            candidate=explanation,
            references=references,
            weights=weights,
            smoothing=smoothing,
            lowercase=lowercase,
        )

    @staticmethod
    def compute_corpus_bleu(
        explanations: Sequence[str],
        references: Sequence[Sequence[str]],
        weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        smoothing: Optional[str] = "method1",
        lowercase: bool = True,
    ) -> float:
        """Computes corpus-level BLEU score across a dataset of explanations."""
        return corpus_bleu(
            candidates=explanations,
            references=references,
            weights=weights,
            smoothing=smoothing,
            lowercase=lowercase,
        )

    @staticmethod
    def compute_semantic_preservation(
        explanation: str,
        semantic_rep: SemanticRepresentation,
    ) -> float:
        """
        Calculates the proportion of salient clinical findings in the semantic representation
        that are accurately conveyed in the generated explanation.
        """
        findings = semantic_rep.clinical_findings
        if not findings:
            return 1.0  # Baseline stability preserved

        preserved_count = 0
        text_lower = explanation.lower()

        for f in findings:
            clean_name = f.lab_name.lower().split("/")[0].split("(")[0].strip()
            # Match either lab name or relevant acronym (e.g. BUN, WBC, INR, PT, PTT)
            acronym_match = False
            if "(" in f.lab_name and ")" in f.lab_name:
                acronym = f.lab_name.split("(")[1].split(")")[0].strip().lower()
                if re.search(rf"\b{re.escape(acronym)}\b", text_lower):
                    acronym_match = True

            name_match = re.search(rf"\b{re.escape(clean_name)}\b", text_lower) or acronym_match

            if name_match:
                # Check directional alignment
                if f.direction == "high":
                    dir_match = any(w in text_lower for w in ["elevated", "high", "increased", "peak"])
                elif f.direction == "low":
                    dir_match = any(w in text_lower for w in ["low", "reduced", "decreased", "depressed"])
                else:
                    dir_match = True

                if dir_match:
                    preserved_count += 1

        return preserved_count / len(findings)

    @staticmethod
    def compute_directional_accuracy(
        explanation: str,
        semantic_rep: SemanticRepresentation,
    ) -> float:
        """Verifies that no lab findings are directionally inverted."""
        findings = semantic_rep.clinical_findings
        if not findings:
            return 1.0

        text_lower = explanation.lower()
        correct_count = 0
        evaluated_count = 0

        for f in findings:
            clean_name = f.lab_name.lower().split("/")[0].split("(")[0].strip()
            name_present = clean_name in text_lower

            if name_present:
                evaluated_count += 1
                if f.direction == "high":
                    contradiction = any(
                        re.search(rf"\b(low|reduced|decreased|depressed)\s+{re.escape(clean_name)}\b", text_lower)
                        for _ in [1]
                    )
                    if not contradiction:
                        correct_count += 1
                elif f.direction == "low":
                    contradiction = any(
                        re.search(rf"\b(high|elevated|increased|escalated)\s+{re.escape(clean_name)}\b", text_lower)
                        for _ in [1]
                    )
                    if not contradiction:
                        correct_count += 1
                else:
                    correct_count += 1

        return correct_count / max(evaluated_count, 1)

    @classmethod
    def evaluate_batch(
        cls,
        explanations: List[str],
        semantic_reps: List[SemanticRepresentation],
        reference_texts: Optional[Sequence[Union[str, Sequence[str]]]] = None,
    ) -> Dict[str, Any]:
        """Runs comprehensive batch evaluation over paired explanations and semantic representations."""
        assert len(explanations) == len(semantic_reps), "Counts must match"

        preservations = []
        directional_accs = []
        uncertainty_scores = []
        violations_count = 0
        valid_count = 0
        all_violations: List[str] = []

        for exp, rep in zip(explanations, semantic_reps):
            val_res = ExplanationValidator.validate(exp, rep)
            if val_res.is_valid:
                valid_count += 1
            else:
                violations_count += 1
                all_violations.extend(val_res.violations)

            p_score = cls.compute_semantic_preservation(exp, rep)
            d_acc = cls.compute_directional_accuracy(exp, rep)

            text_lower = exp.lower()
            has_unc = 1.0 if any(
                w in text_lower for w in ["does not establish a specific diagnosis", "uncertainty", "correlation", "interpret"]
            ) else 0.0

            preservations.append(p_score)
            directional_accs.append(d_acc)
            uncertainty_scores.append(has_unc)

        n = max(len(explanations), 1)
        mean_preservation = float(np.mean(preservations))
        unsupported_claim_rate = float(violations_count / n)
        mean_directional_acc = float(np.mean(directional_accs))
        mean_uncertainty_coverage = float(np.mean(uncertainty_scores))
        pass_rate = float(valid_count / n)

        # Composite Clinical Faithfulness Index (0.0 to 1.0)
        faithfulness_index = float(
            0.35 * mean_preservation
            + 0.25 * mean_directional_acc
            + 0.20 * mean_uncertainty_coverage
            + 0.20 * (1.0 - unsupported_claim_rate)
        )

        res: Dict[str, Any] = {
            "num_evaluated": len(explanations),
            "semantic_preservation_score": round(mean_preservation, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "directional_accuracy": round(mean_directional_acc, 4),
            "uncertainty_coverage": round(mean_uncertainty_coverage, 4),
            "validation_pass_rate": round(pass_rate, 4),
            "clinical_faithfulness_index": round(faithfulness_index, 4),
            "total_violations_flagged": len(all_violations),
            "sample_violations": all_violations[:5],
        }

        if reference_texts is not None and len(reference_texts) == len(explanations):
            corpus_refs = [
                [r] if isinstance(r, str) else list(r)
                for r in reference_texts
            ]
            c_bleu = cls.compute_corpus_bleu(explanations, corpus_refs)
            sent_bleus = [
                cls.compute_sentence_bleu(exp, refs)
                for exp, refs in zip(explanations, corpus_refs)
            ]
            res["corpus_bleu_4"] = round(c_bleu, 4)
            res["mean_sentence_bleu_4"] = round(float(np.mean(sent_bleus)), 4)

        return res
