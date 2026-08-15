"""
dataset.py
==========
Backward-compatible wrapper re-exporting PyTorch Dataset & DataLoader utilities.
"""

from medical_llm.data.dataset import (
    MedicalDataset,
    train_test_split_custom,
    build_vocab,
    get_dataloaders,
    load_dataset_json,
)

__all__ = [
    "MedicalDataset",
    "train_test_split_custom",
    "build_vocab",
    "get_dataloaders",
    "load_dataset_json",
]