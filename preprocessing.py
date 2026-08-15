"""
preprocessing.py
================
Backward-compatible CLI wrapper for medical data preprocessing.
"""

import sys
from medical_llm.data.text_utils import clean_text
from medical_llm.data.preprocessing import run_preprocessing, process_admissions_and_patients, process_labevents, process_clinical_reports


def main():
    run_preprocessing()


if __name__ == "__main__":
    main()
