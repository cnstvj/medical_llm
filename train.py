"""
train.py
========
CLI runner for single model training.

Usage:
  python train.py                # train on real dataset
  python train.py --dummy        # train on dummy dataset
  python train.py --unimodal     # train time-series Transformer only
  python train.py --epochs 15    # custom training epochs
  python train.py --rebuild      # force regenerating dataset artifacts
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import DEFAULT_CONFIG, parse_pipeline_args
from medical_llm.pipeline.trainer import train, prepare_data, evaluate
from medical_llm.utils.logging import log_event

__all__ = ["train", "prepare_data", "evaluate", "log_event", "main"]


def main():
    config, dummy_mode, _ = parse_pipeline_args()
    rebuild = "--rebuild" in sys.argv
    train(dummy_mode=dummy_mode, config=config, force_rebuild=rebuild)


if __name__ == "__main__":
    main()
