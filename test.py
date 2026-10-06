"""
test.py
=======
CLI runner for evaluating saved model checkpoints and running Semantic Communication audits.

Usage:
  python test.py            # evaluate on real dataset
  python test.py --dummy    # evaluate on dummy dataset
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import DEFAULT_CONFIG, parse_pipeline_args
from medical_llm.pipeline.evaluator import (
    evaluate_model,
    compute_metrics,
    print_metrics,
    run_test_evaluation,
)
from medical_llm.pipeline.trainer import prepare_data

__all__ = [
    "evaluate_model",
    "compute_metrics",
    "print_metrics",
    "run_test_evaluation",
    "prepare_data",
    "main",
]


def main():
    config, dummy_mode, _ = parse_pipeline_args()
    prepare_data(dummy_mode=dummy_mode, config=config)
    run_test_evaluation(dummy_mode=dummy_mode, config=config)


if __name__ == "__main__":
    main()
