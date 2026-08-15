# Medical LLM

A modular PyTorch Transformer and NLP pipeline for temporal sequence processing and mortality prediction using MIMIC-III structured medical records.

---

## Package Architecture

The project is structured as a clean Python package under `src/medical_llm`:

```
medical_llm/
├── pyproject.toml                     # Package configuration & dependencies
├── README.md                          # Project documentation
├── src/
│   └── medical_llm/                   # Core Python package
│       ├── __init__.py                # Package initialization
│       ├── config.py                  # Centralized configuration & hyperparameters
│       ├── utils/                     # Utilities package
│       │   ├── __init__.py
│       │   ├── logging.py             # Event logging helper
│       │   └── quantiles.py           # Continuous lab measurement quantization
│       ├── data/                      # Data processing package
│       │   ├── __init__.py
│       │   ├── text_utils.py          # Clinical note normalization
│       │   ├── preprocessing.py       # Data cleaning, merging, and quantization
│       │   ├── sequence_builder.py    # Temporal sequence construction
│       │   └── dataset.py             # PyTorch Dataset, DataLoaders & vocabulary
│       ├── models/                    # Model architecture package
│       │   ├── __init__.py
│       │   └── transformer.py         # PositionalEncoding & MedicalTransformer
│       └── pipeline/                  # Execution pipeline package
│           ├── __init__.py
│           ├── evaluator.py           # Metrics calculation & test inference
│           ├── trainer.py             # Training loop with class-imbalance weighting
│           └── cross_validator.py     # Stratified 5-Fold Cross Validation
├── preprocessing.py                   # Top-level CLI wrapper
├── sequence_builder.py                # Top-level CLI wrapper
├── dataset.py                         # Backward-compatibility wrapper
├── model.py                           # Backward-compatibility wrapper
├── quantile_utils.py                  # Backward-compatibility wrapper
├── train.py                           # CLI training runner
├── test.py                            # CLI evaluation runner
├── cross_validate.py                  # CLI cross-validation runner
├── rerun_pipeline.py                  # Complete end-to-end pipeline runner
└── generate_dummy_data.py             # Synthetic data generator for rapid testing
```

---

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

---

## Usage Guide

### 1. Rapid Dummy Test Run
To test the pipeline without full production MIMIC datasets:

```bash
python generate_dummy_data.py
python rerun_pipeline.py --dummy
```

### 2. Full Training Pipeline
To run pre-processing, sequence building, and cross-validation on production MIMIC-III CSV files:

```bash
python rerun_pipeline.py
```

### 3. Individual CLI Steps
- **Data Preprocessing**: `python preprocessing.py`
- **Sequence Building**: `python sequence_builder.py`
- **Single Model Training**: `python train.py` (or `python train.py --dummy`)
- **5-Fold Cross Validation**: `python cross_validate.py`
- **Model Evaluation**: `python test.py` (or `python test.py --dummy`)

---

## Python API Usage

You can also import and use `medical_llm` components in custom scripts:

```python
from medical_llm.config import PipelineConfig
from medical_llm.models import MedicalTransformer
from medical_llm.data import get_dataloaders
from medical_llm.pipeline import train_fold

# Custom configuration
config = PipelineConfig(d_model=256, nhead=8, epochs=15)

# Load data loaders
train_loader, val_loader, test_loader, vocab = get_dataloaders("sequence_dataset.json", batch_size=config.batch_size)

# Instantiate Transformer model
model = MedicalTransformer(vocab_size=len(vocab), d_model=config.d_model)
```
