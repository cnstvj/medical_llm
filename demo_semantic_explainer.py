"""
Demo Script: NLP-based Semantic Communication System.
Demonstrates the conversion:
Technical tokens + Mortality probability
  -> Clinical Dictionary mapping
  -> Semantic Transformer Encoder
  -> Structured Semantic Representation
  -> Semantic Transformer Decoder
  -> Explanation Validator
  -> Human-readable Clinical Narrative
"""

import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.models.clinical_explainer import ClinicalExplainer
from medical_llm.semantic.metrics import SemanticEvaluationSuite
from medical_llm.semantic.representation import SemanticRepresentation


def main():
    print("=" * 70)
    print("  NLP-BASED SEMANTIC COMMUNICATION SYSTEM (DEMO)")
    print("=" * 70)

    # 1. Technical Input
    lab_tokens = [
        "AGE_70", "GENDER_F", "ADM_EMERGENCY", "[SEP]",
        "LAB_50813_Q9", "TIME_LT1H",   # Lactate: Q9 (top decile), <1 hour gap
        "LAB_50912_Q9", "TIME_1TO6H",  # Creatinine: Q9 (top decile), 1-6 hours gap
        "LAB_50809_Q8",                # Glucose: Q8 (high decile)
    ]
    mortality_prob = 0.82
    clinical_text = "Patient arrived via ED in septic shock with hypotension."

    print("\n1. TECHNICAL INPUT:")
    print("-" * 50)
    print(f"Demographics : AGE_70, GENDER_F, ADM_EMERGENCY")
    print(f"Tokens       : {' + '.join([t for t in lab_tokens if not t.startswith('AGE_') and not t.startswith('GENDER_') and not t.startswith('ADM_')])}")
    print(f"Notes Text   : \"{clinical_text}\"")
    print(f"Model Prob   : {mortality_prob:.2f} ({mortality_prob*100:.1f}% In-Hospital Mortality Risk)")

    # 2. Run Semantic Communication Pipeline
    result = ClinicalExplainer.generate_patient_rationale(
        probability=mortality_prob,
        lab_tokens=lab_tokens,
        clinical_text=clinical_text,
        validate=True,
    )

    # 3. Print Structured Semantic Representation
    print("\n2. STRUCTURED SEMANTIC REPRESENTATION (JSON):")
    print("-" * 50)
    print(json.dumps(result["semantic_representation"], indent=2))

    # 4. Print Natural Language Explanation
    print("\n3. GENERATED HUMAN-READABLE CLINICAL EXPLANATION:")
    print("-" * 50)
    print(f"\"{result['clinical_narrative']}\"")

    # 5. Print Validation Auditor Results
    print("\n4. EXPLANATION VALIDATOR AUDIT:")
    print("-" * 50)
    val = result["validation"]
    status_str = "PASSED (100% Faithful)" if val["is_valid"] else "FLAGGED"
    print(f"Audit Status        : {status_str}")
    print(f"Hallucination Score : {val['hallucination_score']:.2f} (0.00 = clean)")
    print(f"Violations          : {val['violations'] if val['violations'] else 'None (Zero unsupported claims)'}")
    print("Passed Checks       :")
    for chk in val["passed_checks"]:
        print(f"  [x] {chk}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
