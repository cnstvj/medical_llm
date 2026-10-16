"""
Utilities package for logging and continuous data quantization.
"""

from medical_llm.utils.logging import log_event
from medical_llm.utils.quantiles import (
    fit_quantiles,
    apply_quantile,
    apply_quantiles_to_labevents,
    save_quantiles,
    load_quantiles,
)
from medical_llm.utils.audit import LeakageAuditor
from medical_llm.utils.bleu import (
    BLEUResult,
    compute_ngrams,
    modified_precision,
    calculate_brevity_penalty,
    closest_reference_length,
    sentence_bleu,
    sentence_bleu_detailed,
    corpus_bleu,
    corpus_bleu_detailed,
)

__all__ = [
    "log_event",
    "fit_quantiles",
    "apply_quantile",
    "apply_quantiles_to_labevents",
    "save_quantiles",
    "load_quantiles",
    "LeakageAuditor",
    "BLEUResult",
    "compute_ngrams",
    "modified_precision",
    "calculate_brevity_penalty",
    "closest_reference_length",
    "sentence_bleu",
    "sentence_bleu_detailed",
    "corpus_bleu",
    "corpus_bleu_detailed",
]

