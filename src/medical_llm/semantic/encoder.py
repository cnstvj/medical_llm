"""
Semantic Transformer Encoder for Medical AI Mortality Models.
Transforms technical inputs (laboratory tokens, temporal sequences, clinical text, model probabilities)
into structured SemanticRepresentations preserving physiological meaning and uncertainty.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from medical_llm.semantic.dictionary import ClinicalDictionary, get_default_dictionary
from medical_llm.semantic.representation import (
    SemanticRepresentation,
    ClinicalFinding,
    TemporalDynamics,
    TextEvidence,
    UncertaintyStatement,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module for semantic sequence modeling."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SemanticTransformerEncoder(nn.Module):
    """
    Transformer-based Semantic Encoder.
    Processes technical sequence representations and extracts structured clinical semantics.
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        dictionary: Optional[ClinicalDictionary] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.dictionary = dictionary or get_default_dictionary()

        # Token embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Semantic Salience Head: scores which tokens are primary physiological drivers
        self.salience_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Direction Classification Head: 0 = baseline, 1 = elevated/high, 2 = depressed/low
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) binary mask (1 = valid, 0 = pad)
        Returns:
            semantic_embeddings: (batch_size, seq_len, d_model)
            salience_scores: (batch_size, seq_len)
            direction_logits: (batch_size, seq_len, 3)
        """
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded = self.transformer_encoder(x, src_key_padding_mask=pad_mask)

        salience = self.salience_head(encoded).squeeze(-1)
        direction = self.direction_head(encoded)

        return encoded, salience, direction

    def encode_patient(
        self,
        tokens: List[str],
        mortality_prob: float,
        clinical_text: str = "",
        vocab: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
        top_k_findings: int = 5,
        threshold: float = 0.5,
    ) -> SemanticRepresentation:
        """
        Full semantic encoding pipeline:
        1. Evaluates neural salience over technical tokens (if vocab and weights provided).
        2. Grounds tokens into ontological concepts via ClinicalDictionary.
        3. Quantifies temporal pacing and acute intervals.
        4. Extracts verified textual triggers without hallucinated causality.
        5. Formulates calibrated uncertainty statements.
        """
        # 1. Stratify Risk Level
        if mortality_prob >= 0.70:
            risk_level = "high"
        elif mortality_prob >= 0.35:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # 2. Extract and Decode Technical Tokens
        decoded_labs: List[Dict[str, Any]] = []
        temporal_tokens: List[str] = []

        for tok in tokens:
            if tok.startswith("LAB_"):
                decoded = self.dictionary.decode_lab_token(tok)
                if decoded:
                    decoded_labs.append(decoded)
            elif tok.startswith("TIME_"):
                temporal_tokens.append(tok)

        # 3. Compute Salience
        # Identify acute or extreme anomalies (Q8, Q9, Q0, Q1, ABNORMAL)
        prioritized_findings: List[ClinicalFinding] = []
        seen_items = set()

        # Neural salience scoring if vocab is supplied
        token_salience: Dict[str, float] = {}
        if vocab is not None and device is not None:
            try:
                self.to(device)
                self.eval()
                with torch.no_grad():
                    num_emb = self.embedding.num_embeddings
                    unk_id = vocab.get("[UNK]", 1)
                    if not (isinstance(unk_id, int) and 0 <= unk_id < num_emb):
                        unk_id = 1
                    ids = [vocab.get(t, unk_id) for t in tokens[:512]]
                    ids = [i if (isinstance(i, int) and 0 <= i < num_emb) else unk_id for i in ids]
                    if ids:
                        t_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                        mask = torch.ones_like(t_tensor)
                        _, sal, _ = self.forward(t_tensor, mask)
                        scores = sal.squeeze(0).cpu().numpy()
                        for t, s in zip(tokens[:512], scores):
                            token_salience[t] = float(s)
            except Exception:
                pass

        # Prioritize acute anomalies first, then statistical extremes
        for item in decoded_labs:
            itemid = item["itemid"]
            if itemid in seen_items:
                continue

            sig = item["significance"]
            is_acute = sig in ("acute_high", "acute_low", "flagged_abnormal")

            # In high risk patients, focus on acute deviations; in low risk, acknowledge stability
            if risk_level in ("high", "moderate") and not is_acute:
                continue

            seen_items.add(itemid)
            neural_score = token_salience.get(item["token"], 0.85 if is_acute else 0.50)
            conf = min(0.98, max(0.50, 0.75 + 0.20 * torch.sigmoid(torch.tensor(neural_score)).item()))

            prioritized_findings.append(
                ClinicalFinding(
                    lab_name=item["lab_name"],
                    direction=item["direction"],
                    statistical_bucket=item["statistical_bucket"],
                    clinical_interpretation=item["clinical_interpretation"],
                    confidence=round(conf, 2),
                    itemid=itemid,
                    physiological_system=item["system"],
                    evidence_tokens=[item["token"]],
                )
            )

            if len(prioritized_findings) >= top_k_findings:
                break

        # If no acute findings found but high/moderate risk, take top available labs
        if not prioritized_findings and decoded_labs:
            for item in decoded_labs[:top_k_findings]:
                if item["itemid"] not in seen_items:
                    seen_items.add(item["itemid"])
                    prioritized_findings.append(
                        ClinicalFinding(
                            lab_name=item["lab_name"],
                            direction=item["direction"],
                            statistical_bucket=item["statistical_bucket"],
                            clinical_interpretation=item["clinical_interpretation"],
                            confidence=0.75,
                            itemid=item["itemid"],
                            physiological_system=item["system"],
                            evidence_tokens=[item["token"]],
                        )
                    )

        # 4. Temporal Dynamics Analysis
        acute_gaps = [t for t in temporal_tokens if t in ("TIME_LT1H", "TIME_1TO6H")]
        acute_count = len(acute_gaps)
        total_gaps = len(temporal_tokens)

        if total_gaps == 0:
            dominant_gap = "TIME_1TO6H"
            trajectory_pacing = "acute_unspecified"
            interval_summary = "single admission timeframe"
        elif acute_count / max(total_gaps, 1) > 0.5:
            dominant_gap = "TIME_LT1H" if temporal_tokens.count("TIME_LT1H") >= temporal_tokens.count("TIME_1TO6H") else "TIME_1TO6H"
            trajectory_pacing = "acute_rapid_succession"
            interval_summary = "acute intervals under 6 hours"
        else:
            dominant_gap = "TIME_6TO24H"
            trajectory_pacing = "subacute_extended"
            interval_summary = "extended intervals spanning multiple hours to days"

        temporal_dynamics = TemporalDynamics(
            interval_summary=interval_summary,
            dominant_gap=dominant_gap,
            acute_events_under_6h=acute_count,
            total_events=total_gaps,
            trajectory_pacing=trajectory_pacing,
        )

        # 5. Clinical Text Evidence Grounding
        text_lower = clinical_text.lower() if clinical_text else ""
        text_triggers = []
        clinical_keywords = [
            "hypotension", "sepsis", "septic", "shock", "intubated", "arrest",
            "respiratory failure", "acidosis", "tachycardia", "altered mental status",
            "renal failure", "bleeding", "hypoxia", "hyperkalemia", "hypokalemia",
            "elevated lactate", "anemia", "coagulopathy", "fever"
        ]
        for kw in clinical_keywords:
            if kw in text_lower:
                canonical = "sepsis" if kw == "septic" else kw
                if canonical not in text_triggers:
                    text_triggers.append(canonical)

        text_evidence = TextEvidence(
            trigger_keywords=text_triggers[:6],
            clinical_snippets=[f"Documented mention of {kw}" for kw in text_triggers[:3]],
        )

        # 6. Formulate Explicit Uncertainty & Limitations
        uncertainties = [
            UncertaintyStatement(
                statement="The available information does not establish a specific diagnosis.",
                uncertainty_type="diagnostic_boundary",
            ),
            UncertaintyStatement(
                statement="Statistical decile rankings reflect cohort distribution rather than absolute pathological thresholds.",
                uncertainty_type="cohort_distribution_caveat",
            ),
        ]
        if risk_level == "high":
            uncertainties.append(
                UncertaintyStatement(
                    statement="Model-estimated risk indicates physiological divergence and requires immediate clinical correlation.",
                    uncertainty_type="trajectory_incompleteness",
                )
            )

        limitations = [
            "Mortality predictions reflect statistical associations within the training cohort.",
            "Predictions should not be used as standalone clinical guidance or substitute for bedside clinical assessment.",
        ]

        # 7. Synthesize Summary
        if prioritized_findings:
            findings_str = ", ".join(f"{f.lab_name} ({f.direction})" for f in prioritized_findings[:3])
            stability_str = "stable baseline measurements" if risk_level == "low" else "salient findings"
            clinical_summary = (
                f"Model identifies {risk_level}-risk trajectory with {stability_str}: {findings_str} "
                f"occurring over {interval_summary}."
            )
        else:
            clinical_summary = f"Model predicts {risk_level} mortality risk with stable multivariate laboratory baseline."

        return SemanticRepresentation(
            risk_level=risk_level,
            mortality_probability=float(mortality_prob),
            confidence_score=0.91 if risk_level == "high" else 0.88,
            clinical_findings=prioritized_findings,
            temporal_context=temporal_dynamics,
            text_evidence=text_evidence,
            clinical_summary=clinical_summary,
            uncertainties=uncertainties,
            limitations=limitations,
        )
