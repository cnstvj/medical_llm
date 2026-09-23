"""
Clinical text normalization and NLP helper utilities.
"""

import re
from typing import Any
import pandas as pd


def clean_text(text: Any) -> str:
    """Normalizes clinical text, stripping direct patient identifiers, admission IDs, exact calendar dates, and timestamps while preserving medical findings and clinical negation semantics."""
    if pd.isnull(text):
        return ""
    text = str(text)

    # 1. Strip explicit patient and admission identifier headers (e.g., Patient ID: 10076, Hadm ID: 12345)
    text = re.sub(r"(?i)\b(patient|subject|hadm|admission|record)\s*(id|_id)?\s*:\s*\d+\b", " ", text)

    # 2. Strip explicit date and time headers (e.g., Admission Date: 2107-03-21, Admission Time: 17:50:00)
    text = re.sub(r"(?i)\b(admission|discharge|chart|dob|birth|death|report)?\s*(date|time|datetime)\s*:\s*[\d\-/:apm\s]+(?=(\n|\r|$|[A-Z][a-z]+:))", " ", text)

    # 3. Strip remaining exact dates (e.g. YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY)
    text = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", text)

    # 4. Strip remaining standalone time strings (e.g. 17:50:00, 06:10)
    text = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", " ", text)

    # 5. Convert to lowercase and preserve standard medical characters/punctuation (e.g. hyphens, periods for clinical sentences)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\-\/]", " ", text)

    # 6. Normalize whitespace
    return " ".join(text.split())
