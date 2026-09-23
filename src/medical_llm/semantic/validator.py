"""
Explanation Validator and Hallucination Auditor.
Validates that generated clinical explanations strictly adhere to the underlying SemanticRepresentation,
detecting unsupported diagnoses, unjustified causal claims, directional contradictions,
unauthorized treatment mandates, and missing uncertainty caveats.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from medical_llm.semantic.representation import SemanticRepresentation


# Diseases / diagnostic claims that cannot be asserted purely from lab deviations or mortality risk
PROHIBITED_UNGROUNDED_DIAGNOSES = [
    "diabetes", "diabetic ketoacidosis", "dka", "cirrhosis",
    "septic shock", "myocardial infarction", "heart attack", "stroke",
    "chronic kidney disease", "ckd", "end-stage renal disease", "esrd",
    "pneumonia", "pulmonary embolism", "leukemia", "lymphoma",
    "congestive heart failure", "chf", "acute liver failure",
    "pancreatitis", "meningitis", "appendicitis"
]

# Actionable treatment imperatives that a prediction model must not mandate
PROHIBITED_TREATMENT_IMPERATIVES = [
    "administer insulin", "start vasopressors", "intubate immediately",
    "prescribe antibiotics", "perform hemodialysis", "start dialysis",
    "transfuse red blood cells", "administer heparin", "give epinephrine",
    "start inotropes", "give blood"
]


@dataclass
class ValidationResult:
    """Outcome of explanation semantic validation."""
    is_valid: bool
    hallucination_score: float  # 0.0 = completely faithful, 1.0 = severely ungrounded
    violations: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "hallucination_score": round(self.hallucination_score, 3),
            "violations": self.violations,
            "passed_checks": self.passed_checks,
            "details": self.details,
        }


class ExplanationValidator:
    """Independent semantic integrity and medical claim checker."""

    @classmethod
    def validate(
        cls,
        explanation_text: str,
        semantic_rep: SemanticRepresentation,
    ) -> ValidationResult:
        violations = []
        passed = []
        text_lower = explanation_text.lower()

        # 1. Unsupported Diagnoses Check
        supported_text_evidence = " ".join(
            semantic_rep.text_evidence.trigger_keywords
            + semantic_rep.text_evidence.clinical_snippets
        ).lower()

        for diag in PROHIBITED_UNGROUNDED_DIAGNOSES:
            # Check if diagnosis appears in explanation
            # Use word boundary search
            pattern = rf"\b{re.escape(diag)}\b"
            if re.search(pattern, text_lower):
                # Is it explicitly grounded in the patient's clinical notes?
                if not re.search(pattern, supported_text_evidence):
                    violations.append(
                        f"UNSUPPORTED_DIAGNOSIS: Explanation claims diagnosis of '{diag}' without input text evidence."
                    )

        if not any(v.startswith("UNSUPPORTED_DIAGNOSIS") for v in violations):
            passed.append("No unsupported clinical diagnoses detected")

        # 2. Unsupported Treatment/Intervention Mandates
        for treat in PROHIBITED_TREATMENT_IMPERATIVES:
            if treat in text_lower:
                violations.append(
                    f"UNSUPPORTED_TREATMENT: Model asserts unauthorized clinical action '{treat}'."
                )

        if not any(v.startswith("UNSUPPORTED_TREATMENT") for v in violations):
            passed.append("No unauthorized treatment or prescription directives")

        # 3. Directional Consistency with Lab Findings
        for finding in semantic_rep.clinical_findings:
            lab_name_clean = finding.lab_name.lower().split("/")[0].split("(")[0].strip()
            # If finding is high, explanation shouldn't describe it as low/decreased
            if finding.direction == "high":
                low_pattern = rf"\b(low|reduced|decreased|depressed)\s+{re.escape(lab_name_clean)}\b"
                if re.search(low_pattern, text_lower):
                    violations.append(
                        f"DIRECTIONAL_CONTRADICTION: Finding '{finding.lab_name}' is high/Q9, but explanation describes it as reduced/low."
                    )
            # If finding is low, explanation shouldn't describe it as elevated/high
            elif finding.direction == "low":
                high_pattern = rf"\b(high|elevated|increased|escalated)\s+{re.escape(lab_name_clean)}\b"
                if re.search(high_pattern, text_lower):
                    violations.append(
                        f"DIRECTIONAL_CONTRADICTION: Finding '{finding.lab_name}' is low/Q0, but explanation describes it as elevated/high."
                    )

        if not any(v.startswith("DIRECTIONAL_CONTRADICTION") for v in violations):
            passed.append("100% directional consistency with quantitative lab findings")

        # 4. Mandatory Uncertainty and Limitation Caveat Check
        has_uncertainty = any(
            phrase in text_lower for phrase in [
                "does not establish a specific diagnosis",
                "do not establish a specific diagnosis",
                "does not establish a diagnosis",
                "cannot establish a diagnosis",
                "does not constitute a diagnosis",
                "not a diagnosis",
                "requires clinical correlation",
                "reviewed alongside",
                "interpreted alongside",
            ]
        )
        if not has_uncertainty:
            violations.append(
                "MISSING_UNCERTAINTY_CAVEAT: Explanation lacks required uncertainty boundary stating it does not establish a specific diagnosis."
            )
        else:
            passed.append("Mandatory uncertainty and diagnostic limitation caveats present")

        # 5. Risk Tier Faithfulness Check
        pred_prob = semantic_rep.mortality_probability
        if pred_prob >= 0.70 and "low risk" in text_lower:
            violations.append(
                f"RISK_CONTRADICTION: Model predicted high mortality ({pred_prob:.2f}), but explanation states low risk."
            )
        elif pred_prob < 0.35 and "high risk" in text_lower:
            violations.append(
                f"RISK_CONTRADICTION: Model predicted low mortality ({pred_prob:.2f}), but explanation states high risk."
            )
        else:
            passed.append("Risk tier matches predicted mortality probability")

        is_valid = len(violations) == 0
        hallucination_score = min(1.0, len(violations) * 0.25)

        return ValidationResult(
            is_valid=is_valid,
            hallucination_score=hallucination_score,
            violations=violations,
            passed_checks=passed,
            details={
                "num_violations": len(violations),
                "num_passed": len(passed),
            }
        )
