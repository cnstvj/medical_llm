"""
sequence_builder.py
===================
Backward-compatible CLI wrapper for temporal sequence building.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import DEFAULT_CONFIG
from medical_llm.data.sequence_builder import (
    time_gap_token,
    build_temporal_sequences,
    run_sequence_builder,
)

EXPERIMENT_MODE = DEFAULT_CONFIG.experiment_mode


def main():
    run_sequence_builder(experiment_mode=EXPERIMENT_MODE)


if __name__ == "__main__":
    main()