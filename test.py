"""
test.py
=======
Backward-compatible CLI wrapper for evaluating saved fold models.

Usage:
  python test.py            # evaluate on real dataset
  python test.py --dummy    # evaluate on dummy dataset
"""

import sys
import os
from medical_llm.pipeline.evaluator import (
    evaluate_model,
    compute_metrics,
    print_metrics,
    run_test_evaluation,
)
from medical_llm.pipeline.trainer import prepare_data

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    dummy_mode = "--dummy" in sys.argv
    prepare_data(dummy_mode=dummy_mode)
    run_test_evaluation(dummy_mode=dummy_mode)


if __name__ == "__main__":
    main()
