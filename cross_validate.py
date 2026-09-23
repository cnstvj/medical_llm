"""
cross_validate.py
=================
Backward-compatible CLI wrapper for 5-fold cross validation.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.pipeline.cross_validator import run_cross_validation, train_fold, evaluate_fold as evaluate


def main():
    run_cross_validation()


if __name__ == "__main__":
    main()