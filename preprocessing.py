"""
preprocessing.py
================
Backward-compatible CLI wrapper for medical data preprocessing.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.data.text_utils import clean_text
from medical_llm.data.preprocessing import run_preprocessing, process_admissions_and_patients, process_labevents, process_clinical_reports


def main():
    run_preprocessing()


if __name__ == "__main__":
    main()
