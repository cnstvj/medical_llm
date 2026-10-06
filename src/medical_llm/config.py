"""
Centralized Configuration for Medical LLM pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


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


def parse_pipeline_args(
    args: Optional[List[str]] = None,
    default_config: Optional[PipelineConfig] = None,
) -> Tuple[PipelineConfig, bool, Optional[str]]:
    """
    Parses standard command-line flags and returns (config, dummy_mode, run_label).
    Supports --dummy, --unimodal, --multimodal, --mode, --epochs, --batch-size, --lr, --patience, --splits, --label.
    """
    import argparse
    import sys

    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dummy", action="store_true", default=False, help="Run in dummy/synthetic data mode")
    parser.add_argument("--unimodal", action="store_true", default=False, help="Disable multi-modal ClinicalBERT notes stream")
    parser.add_argument("--multimodal", action="store_true", default=False, help="Enable multi-modal ClinicalBERT notes stream")
    parser.add_argument("--mode", type=str, default=None, choices=["FULL", "DEMOGRAPHICS", "LABS", "TEMPORAL", "NOTES"])
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--splits", type=int, default=None, help="Number of CV splits")
    parser.add_argument("--label", "--tag", dest="label", type=str, default=None, help="Descriptive label for run logs")
    parser.add_argument("--rebuild", action="store_true", default=False, help="Force rebuilding preprocessed artifacts")

    parsed, _ = parser.parse_known_args(args)

    base = default_config or PipelineConfig()
    cfg = PipelineConfig(
        experiment_mode=parsed.mode or base.experiment_mode,
        log_file=base.log_file,
        quantiles_file=base.quantiles_file,
        sequence_dataset_file=base.sequence_dataset_file,
        processed_admissions_file=base.processed_admissions_file,
        processed_labevents_file=base.processed_labevents_file,
        processed_reports_file=base.processed_reports_file,
        admissions_csv=base.admissions_csv,
        patients_csv=base.patients_csv,
        labevents_csv=base.labevents_csv,
        d_labitems_csv=base.d_labitems_csv,
        reports_csv=base.reports_csv,
        d_model=base.d_model,
        nhead=base.nhead,
        num_layers=base.num_layers,
        dim_feedforward=base.dim_feedforward,
        dropout=base.dropout,
        max_len=base.max_len,
        use_multimodal_nlp=(not parsed.unimodal) if parsed.unimodal else (True if parsed.multimodal else base.use_multimodal_nlp),
        pretrained_text_model=base.pretrained_text_model,
        text_max_len=base.text_max_len,
        text_embedding_dim=base.text_embedding_dim,
        fusion_type=base.fusion_type,
        freeze_text_backbone=base.freeze_text_backbone,
        text_learning_rate=base.text_learning_rate,
        batch_size=parsed.batch_size or base.batch_size,
        learning_rate=parsed.lr or base.learning_rate,
        weight_decay=base.weight_decay,
        epochs=parsed.epochs or base.epochs,
        max_epochs=parsed.epochs or base.max_epochs,
        patience=parsed.patience or base.patience,
        n_splits=parsed.splits or base.n_splits,
        random_seed=base.random_seed,
    )

    return cfg, parsed.dummy, parsed.label
