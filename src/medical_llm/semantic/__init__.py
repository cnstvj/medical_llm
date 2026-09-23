"""
Semantic Communication System for Clinical AI Explanations.
"""

from medical_llm.semantic.dictionary import ClinicalDictionary, get_default_dictionary
from medical_llm.semantic.representation import (
    SemanticRepresentation,
    ClinicalFinding,
    TemporalDynamics,
    TextEvidence,
    UncertaintyStatement,
)
from medical_llm.semantic.encoder import SemanticTransformerEncoder
from medical_llm.semantic.decoder import SemanticTransformerDecoder
from medical_llm.semantic.validator import ExplanationValidator, ValidationResult
from medical_llm.semantic.metrics import SemanticEvaluationSuite

__all__ = [
    "ClinicalDictionary",
    "get_default_dictionary",
    "SemanticRepresentation",
    "ClinicalFinding",
    "TemporalDynamics",
    "TextEvidence",
    "UncertaintyStatement",
    "SemanticTransformerEncoder",
    "SemanticTransformerDecoder",
    "ExplanationValidator",
    "ValidationResult",
    "SemanticEvaluationSuite",
]
