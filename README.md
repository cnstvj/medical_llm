# Medical LLM

A modular PyTorch Transformer and clinical NLP pipeline for in-hospital mortality prediction using MIMIC-III electronic health records. The system integrates structured longitudinal laboratory trajectories with unstructured clinical notes via `Bio_ClinicalBERT`, enforces a 4-layer data leakage shield, and provides an explainable Semantic Communication layer with an automated hallucination auditor.

---

## 1. System Architecture

```
medical_llm/
├── pyproject.toml                         # Package configuration & dependencies
├── README.md                              # Project documentation
├── run.bat                                # Windows execution runner
│
├── data/                                  # All CSV data datasets
│   ├── real/                              # Real MIMIC-III cohort datasets
│   │   ├── admissions.csv                 # Hospital encounters & outcome flags
│   │   ├── patients.csv                   # Demographics & birth dates
│   │   ├── labevents.csv                  # Longitudinal laboratory observations
│   │   ├── d_labitems.csv                 # Lab item metadata & analyte labels
│   │   └── structured_medical_records.csv # Clinical notes & triage reports
│   └── dummy/                             # Synthetic dummy datasets for rapid local testing
│       ├── admissions_dummy.csv
│       ├── patients_dummy.csv
│       ├── labevents_dummy.csv
│       ├── d_labitems_dummy.csv
│       ├── structured_medical_records_dummy.csv
│       └── chartevents_dummy.csv
│
├── CLI Entry Points
│   ├── rerun_pipeline.py                  # End-to-end pipeline orchestrator
│   ├── train.py                           # CLI: single model training
│   ├── cross_validate.py                  # CLI: 5-fold stratified cross-validation
│   └── test.py                            # CLI: checkpoint evaluation & audit
│
├── synthetic/                             # Synthetic data & demo utilities
│   ├── generate_dummy_data.py             # Synthetic cohort generator for rapid testing
│   └── demo_semantic_explainer.py         # NLP Semantic Explainer demonstration
│
├── docs/                                  # Documentation
│   └── pipeline_architecture.md           # In-depth step-by-step technical architecture
│
├── src/medical_llm/                       # Canonical Python Package
│   ├── __init__.py                        # Package initialization
│   ├── config.py                          # Hyperparameters, dataclass config & CLI parsing
│   │
│   ├── data/                              # Data Processing Subpackage
│   │   ├── __init__.py
│   │   ├── text_utils.py                  # Clinical note identifier & date sanitization
│   │   ├── preprocessing.py               # Patient alignment, HIPAA age capping, leak-free quantiles
│   │   ├── sequence_builder.py            # Irregular interval bucketing & sequence generation
│   │   └── dataset.py                     # PyTorch Datasets & patient-level group splitting
│   │
│   ├── models/                            # Model Architectures
│   │   ├── __init__.py
│   │   ├── transformer.py                 # Sinusoidal PositionalEncoding & Pre-LN MedicalTransformer
│   │   ├── multimodal_transformer.py      # ClinicalTextEncoder (Bio_ClinicalBERT) + Gated Fusion
│   │   └── clinical_explainer.py          # ClinicalExplainer rationale synthesis coordinator
│   │
│   ├── semantic/                          # Semantic Communication Subpackage
│   │   ├── __init__.py
│   │   ├── representation.py              # SemanticRepresentation structured dataclass model
│   │   ├── dictionary.py                  # Clinical knowledge dictionary & analyte profiles
│   │   ├── encoder.py                     # SemanticTransformerEncoder & salience extraction
│   │   ├── decoder.py                     # SemanticTransformerDecoder narrative generator
│   │   ├── validator.py                   # ExplanationValidator hallucination & safety auditor
│   │   └── metrics.py                     # Semantic preservation & clinical faithfulness suite
│   │
│   ├── pipeline/                          # Pipeline Execution Subpackage
│   │   ├── __init__.py
│   │   ├── trainer.py                     # Training loop with class-imbalance weighting
│   │   ├── cross_validator.py             # Leak-free 5-Fold StratifiedGroupKFold engine
│   │   └── evaluator.py                   # Multi-metric threshold-tuning & test evaluation
│   │
│   └── utils/                             # Utilities
│       ├── __init__.py
│       ├── audit.py                       # Automated 15-point data leakage and invariant auditor
│       ├── logging.py                     # Centralized event logging helper
│       └── quantiles.py                   # Continuous measurement quantization & serialization
│
└── tests/                                 # Verification Test Suite (32 Unit Tests)
    ├── test_patient_leakage.py            # Verifies 0% patient overlap across splits
    ├── test_quantile_leakage.py           # Verifies quantiles fitted strictly on train sets
    ├── test_temporal_leakage.py           # Verifies zero post-discharge lookahead events
    ├── test_identifier_leakage.py         # Verifies stripping of Subject/Admission IDs & dates
    ├── test_leakage_audit.py              # Automated 15-point leakage auditor verification
    ├── test_semantic_encoder.py           # Verifies representation construction & direction
    ├── test_semantic_decoder.py           # Verifies faithful narrative generation
    ├── test_semantic_validator.py         # Verifies hallucination & unauthorized directive detection
    └── test_semantic_preservation.py      # Verifies end-to-end semantic fidelity
```

---

## 2. Multi-Layer Data Leakage Shield

Medical predictive AI is exceptionally vulnerable to data leakage. This system enforces four strict defense layers:

1. **Patient Leakage**: Enforced by `StratifiedGroupKFold` and `train_test_split_by_subject`. All hospital admissions for a given `subject_id` are grouped strictly into the same split (0% patient overlap).
2. **Quantile Leakage**: Lab decile cutoffs ($Q0-Q9$) are fitted **strictly** on training split lab events (`fit_quantiles()`), serialized to disk, and statically applied to validation and test folds (`apply_quantiles_to_labevents()`).
3. **Temporal Lookahead Leakage**: Measurements recorded after the prediction cutoff ($t > \min(\text{dischtime}, \text{deathtime})$) or earlier than 24 hours prior to admission are filtered out during ingestion.
4. **Identifier Leakage**: `clean_text()` strips patient IDs, admission numbers, and calendar timestamps from clinical notes while preserving clinical findings and medical negation.

---

## 3. Installation

Install dependencies and the `medical_llm` package in editable mode:

```bash
pip install -e .
```

Requirements:
- Python $\ge$ 3.8
- PyTorch
- Transformers
- Pandas, NumPy, Scikit-Learn

---

## 4. Usage Guide

### Rapid Synthetic Test Run (No Production MIMIC Required)
```bash
python synthetic/generate_dummy_data.py
python rerun_pipeline.py --dummy
```
To run rapidly on CPU without loading heavy BERT weights:
```bash
python rerun_pipeline.py --dummy --unimodal
```

### Full Production Pipeline
To run preprocessing, sequence building, and 5-fold cross-validation on production MIMIC-III CSV files:
```bash
python rerun_pipeline.py --label "Production MIMIC-III Full Run"
```

### Run Semantic Communication & Explanation Demo
Demonstrates the conversion of technical model outputs and decile tokens into a structured semantic representation and clinician-readable narrative with automated safety validation:
```bash
python synthetic/demo_semantic_explainer.py
```

### Individual CLI Steps
Every runner supports common flags (`--dummy`, `--unimodal`, `--multimodal`, `--mode <MODE>`, `--epochs <N>`):

- **Single Model Training**: `python train.py` (or `python train.py --dummy`)
- **5-Fold Cross Validation**: `python cross_validate.py` (or `python cross_validate.py --dummy`)
- **Model Evaluation**: `python test.py` (or `python test.py --dummy`)

---

## 5. Running the Test Suite

Execute all unit and integration tests across data leakage shields, neural models, and the semantic communication system:

```bash
pytest -v
```

---

## 6. Python API Usage

```python
from medical_llm.config import PipelineConfig
from medical_llm.models import MedicalTransformer, ClinicalExplainer
from medical_llm.data import get_dataloaders
from medical_llm.pipeline import run_cross_validation

# 1. Custom Configuration
config = PipelineConfig(d_model=128, nhead=4, epochs=10)

# 2. Run Leak-Free 5-Fold Cross Validation
results = run_cross_validation(config=config, dummy_mode=True)
print(f"Mean AUROC: {results['mean_auroc']:.4f}")

# 3. Generate Clinical Explanation for a Patient Trajectory
rationale = ClinicalExplainer.generate_patient_rationale(
    probability=0.82,
    lab_tokens=["LAB_50813_Q9", "TIME_LT1H", "LAB_50912_Q9"],
    clinical_text="Patient arrived via ED with hypotension.",
    validate=True,
)
print("Narrative:", rationale["clinical_narrative"])
print("Audit Status:", "PASSED" if rationale["validation"]["is_valid"] else "FLAGGED")
```

---

## 7. Documentation

For a comprehensive step-by-step architectural deep-dive, see [docs/pipeline_architecture.md](docs/pipeline_architecture.md).
