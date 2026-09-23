"""
Clinical Explainer module for generating structured, clinician-readable risk rationales.
Combines calibrated mortality probabilities with Semantic Transformer Encoder,
Semantic Transformer Decoder, and Explanation Validator.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch

from medical_llm.semantic.dictionary import ClinicalDictionary, get_default_dictionary
from medical_llm.semantic.representation import SemanticRepresentation
from medical_llm.semantic.encoder import SemanticTransformerEncoder
from medical_llm.semantic.decoder import SemanticTransformerDecoder
from medical_llm.semantic.validator import ExplanationValidator, ValidationResult


class ClinicalExplainer:
    """
    NLP-based Semantic Communication Explainer.
    Mediates between medical AI technical representations and human-readable clinical meaning.
    """

    _encoder: Optional[SemanticTransformerEncoder] = None
    _decoder: Optional[SemanticTransformerDecoder] = None
    _dictionary: Optional[ClinicalDictionary] = None

    @classmethod
    def get_components(cls) -> Tuple[SemanticTransformerEncoder, SemanticTransformerDecoder, ClinicalDictionary]:
        if cls._dictionary is None:
            cls._dictionary = get_default_dictionary()
        if cls._encoder is None:
            cls._encoder = SemanticTransformerEncoder(dictionary=cls._dictionary)
        if cls._decoder is None:
            cls._decoder = SemanticTransformerDecoder()
        return cls._encoder, cls._decoder, cls._dictionary

    @staticmethod
    def stratify_risk(prob: float) -> str:
        if prob >= 0.70:
            return "CRITICAL / HIGH RISK"
        elif prob >= 0.35:
            return "MODERATE / ELEVATED RISK"
        else:
            return "LOW / STABLE RISK"

    @classmethod
    def generate_patient_rationale(
        cls,
        probability: float,
        lab_tokens: List[str],
        clinical_text: str = "",
        threshold: float = 0.5,
        vocab: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Synthesizes a structured clinical rationale for a patient prediction using the
        Semantic Communication Pipeline:
        Technical Tokens -> Semantic Transformer Encoder -> SemanticRepresentation
        -> Semantic Transformer Decoder -> Explanation Validator -> Clinician Explanation.
        """
        encoder, decoder, _ = cls.get_components()

        # 1. Semantic Transformer Encoder produces SemanticRepresentation
        semantic_rep = encoder.encode_patient(
            tokens=lab_tokens,
            mortality_prob=probability,
            clinical_text=clinical_text,
            vocab=vocab,
            device=device,
            threshold=threshold,
        )

        # 2. Semantic Transformer Decoder synthesizes Explanation Narrative
        decoded_output = decoder.decode_to_narrative(semantic_rep, mode="hybrid")
        narrative = decoded_output["full_explanation"]

        # 3. Explanation Validator verifies semantic preservation & absence of unsupported claims
        validation_res = None
        if validate:
            validation_res = ExplanationValidator.validate(narrative, semantic_rep)

        tier = cls.stratify_risk(probability)

        # Extract acute lab anomalies formatted for backward compatibility
        acute_labs = []
        for f in semantic_rep.clinical_findings:
            acute_labs.append(f"{f.lab_name} ({f.direction.upper()}, {f.statistical_bucket})")

        return {
            # Backward-compatible fields
            "predicted_risk": float(probability),
            "risk_tier": tier,
            "acute_lab_anomalies": acute_labs,
            "text_clinical_triggers": semantic_rep.text_evidence.trigger_keywords,
            "clinical_narrative": narrative,
            # Rich Semantic Communication System fields
            "semantic_representation": semantic_rep.to_dict(),
            "validation": validation_res.to_dict() if validation_res else None,
            "sections": decoded_output.get("sections", {}),
            "clinical_summary": semantic_rep.clinical_summary,
            "uncertainty_statements": [u.to_dict() for u in semantic_rep.uncertainties],
        }
