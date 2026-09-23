"""
Centralized Configuration for Medical LLM pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    # Experiment modes: "FULL", "DEMOGRAPHICS", "LABS", "TEMPORAL", "NOTES"
    experiment_mode: str = "FULL"

    # Files & Paths
    log_file: str = "log.txt"
    quantiles_file: str = "quantiles.pkl"
    sequence_dataset_file: str = "sequence_dataset.json"

    processed_admissions_file: str = "processed_admissions.csv"
    processed_labevents_file: str = "processed_labevents.csv"
    processed_reports_file: str = "processed_reports.csv"

    # Data CSV names (with optional _DUMMY suffix when in dummy mode)
    admissions_csv: str = "ADMISSIONS.csv"
    patients_csv: str = "PATIENTS.csv"
    labevents_csv: str = "LABEVENTS.csv"
    d_labitems_csv: str = "D_LABITEMS.csv"
    reports_csv: str = "structured_medical_records.csv"

    # Training & Model Parameters
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_len: int = 512

    # Multi-Modal NLP Parameters
    use_multimodal_nlp: bool = True
    pretrained_text_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_len: int = 256
    text_embedding_dim: int = 768
    fusion_type: str = "gated"  # "concat", "gated", "cross_attention"
    freeze_text_backbone: bool = True
    text_learning_rate: float = 2e-5

    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    max_epochs: int = 30
    patience: int = 4
    n_splits: int = 5
    random_seed: int = 42

    def get_csv_name(self, base_name: str, dummy_mode: bool = False) -> str:
        """Returns CSV filename according to dummy mode flag."""
        if not dummy_mode:
            return base_name
        name, ext = os.path.splitext(base_name)
        return f"{name}_DUMMY{ext}"


DEFAULT_CONFIG = PipelineConfig()
