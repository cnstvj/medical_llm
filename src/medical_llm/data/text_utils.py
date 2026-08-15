"""
Clinical text normalization and NLP helper utilities.
"""

import re
from typing import Any
import pandas as pd


def clean_text(text: Any) -> str:
    """Normalizes clinical text by converting to lowercase, removing non-alphanumeric characters, and removing excess whitespace."""
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())
