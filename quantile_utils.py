"""
quantile_utils.py
=================
Backward-compatible wrapper importing quantile processing functionality from medical_llm package.
"""

from medical_llm.utils.quantiles import (
    fit_quantiles,
    apply_quantile,
    save_quantiles,
    load_quantiles,
)

__all__ = ["fit_quantiles", "apply_quantile", "save_quantiles", "load_quantiles"]