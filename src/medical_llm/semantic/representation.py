"""
Structured Semantic Representation Data Model.
Preserves clinical meaning, physiological significance, temporal context,
and explicit uncertainty boundaries.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ClinicalFinding:
    """Represents a single decoded physiological or laboratory finding."""
    lab_name: str
    direction: str  # "high", "low", "baseline", "abnormal", "normal"
    statistical_bucket: str  # e.g. "Q9", "NORMAL"
    clinical_interpretation: str
    confidence: float = 0.90
    itemid: Optional[int] = None
    physiological_system: str = "General"
    evidence_tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalDynamics:
    """Represents temporal pacing, interval frequency, and acute clustering."""
    interval_summary: str = "acute"
    dominant_gap: str = "TIME_LT1H"
    acute_events_under_6h: int = 0
    total_events: int = 0
    trajectory_pacing: str = "acute_frequent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextEvidence:
    """Represents salient textual triggers grounded in clinical notes."""
    trigger_keywords: List[str] = field(default_factory=list)
    clinical_snippets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyStatement:
    """Explicitly articulated uncertainty boundary or caveat."""
    statement: str
    uncertainty_type: str = "diagnostic_boundary"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticRepresentation:
    """
    Complete structured semantic representation mediating between the technical medical AI
    model and natural language clinical explanation.
    """
    risk_level: str  # "high", "moderate", "low"
    mortality_probability: float
    confidence_score: float
    clinical_findings: List[ClinicalFinding] = field(default_factory=list)
    temporal_context: TemporalDynamics = field(default_factory=TemporalDynamics)
    text_evidence: TextEvidence = field(default_factory=TextEvidence)
    clinical_summary: str = ""
    uncertainties: List[UncertaintyStatement] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "mortality_probability": round(float(self.mortality_probability), 4),
            "confidence_score": round(float(self.confidence_score), 4),
            "clinical_findings": [f.to_dict() for f in self.clinical_findings],
            "temporal_context": self.temporal_context.to_dict(),
            "text_evidence": self.text_evidence.to_dict(),
            "clinical_summary": self.clinical_summary,
            "uncertainties": [u.to_dict() for u in self.uncertainties],
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticRepresentation":
        findings = [
            ClinicalFinding(**f) if isinstance(f, dict) else f
            for f in d.get("clinical_findings", [])
        ]
        t_dict = d.get("temporal_context", {})
        temporal = TemporalDynamics(**t_dict) if isinstance(t_dict, dict) else t_dict

        te_dict = d.get("text_evidence", {})
        text_ev = TextEvidence(**te_dict) if isinstance(te_dict, dict) else te_dict

        uncertainties = [
            UncertaintyStatement(**u) if isinstance(u, dict) else u
            for u in d.get("uncertainties", [])
        ]

        return cls(
            risk_level=d.get("risk_level", "low"),
            mortality_probability=float(d.get("mortality_probability", 0.0)),
            confidence_score=float(d.get("confidence_score", 0.9)),
            clinical_findings=findings,
            temporal_context=temporal,
            text_evidence=text_ev,
            clinical_summary=d.get("clinical_summary", ""),
            uncertainties=uncertainties,
            limitations=d.get("limitations", []),
        )
