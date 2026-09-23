"""
train.py
========
Backward-compatible CLI wrapper for model training.

Usage:
  python train.py            # train on real dataset
  python train.py --dummy    # train on dummy dataset
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.pipeline.trainer import train, prepare_data, evaluate
from medical_llm.utils.logging import log_event


def main():
    dummy_mode = "--dummy" in sys.argv
    train(dummy_mode=dummy_mode)


if __name__ == "__main__":
    main()
