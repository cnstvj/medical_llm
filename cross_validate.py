"""
cross_validate.py
=================
CLI runner for 5-Fold Stratified Group Cross-Validation.

Usage:
  python cross_validate.py                # 5-fold CV on real dataset
  python cross_validate.py --dummy        # 5-fold CV on dummy dataset
  python cross_validate.py --unimodal     # Run time-series Transformer only
  python cross_validate.py --mode LABS    # Evaluate on specific ablation subset
  python cross_validate.py --label "Tag"  # Tag this experiment in log.txt
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import DEFAULT_CONFIG, parse_pipeline_args
from medical_llm.pipeline.cross_validator import run_cross_validation, train_fold, evaluate_fold as evaluate

__all__ = ["run_cross_validation", "train_fold", "evaluate", "main"]


def main():
    config, dummy_mode, run_label = parse_pipeline_args()
    run_cross_validation(config=config, dummy_mode=dummy_mode, run_label=run_label)


if __name__ == "__main__":
    main()