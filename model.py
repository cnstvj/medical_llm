"""
model.py
========
Backward-compatible wrapper re-exporting PyTorch Transformer model definitions.
"""

from medical_llm.models.transformer import PositionalEncoding, MedicalTransformer

__all__ = ["PositionalEncoding", "MedicalTransformer"]
