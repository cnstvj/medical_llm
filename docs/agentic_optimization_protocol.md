# Antigravity Agentic Optimization & Iterative Improvement Protocol

## 1. Executive Vision & Objectives
This document defines the official engineering and scientific protocol for utilizing Google Antigravity's autonomous agentic features (`/boost`, subagents, tool pipelines, and validation loops) to iteratively optimize the Multimodal Medical Transformer model. 

The primary goal is to **peak model performance and depth of learning using only the given MIMIC-III subset dataset** while:
1. **Preserving 100% Scientific and Technical Validation**: Strict adherence to zero patient leakage, dynamic fold-isolated quantile fitting, and 5-fold Stratified Group Cross-Validation.
2. **Elevating to Modern AI Standards**: Enabling fluent Natural Language Input (clinical questions, patient EHR trajectory queries) and Natural Language Output (uncertainty-calibrated risk tiering, faithful physiological explanations, and audited clinician narratives).
3. **Preserving Codebase Stability**: Building upon the existing architecture without unnecessary refactoring or variable renames.

---

## 2. Core Architecture State & Capabilities
The system builds directly upon existing modules in `src/medical_llm`:

| Module | Core Responsibility | Existing Implementation |
|---|---|---|
| **Continuous Time** | Irregular timestamp interval modeling | `Time2Vec` (Linear trend + Fourier harmonics) |
| **Multimodal Backbone** | Structured labs + Unstructured notes | `MultiModalMedicalTransformer` with Pre-LN & Bio_ClinicalBERT |
| **Adaptation** | Parameter-efficient fine-tuning | LoRA adapters on BERT attention (`query`, `value`) |
| **Fusion Layer** | Token-level interaction | `BidirectionalCrossAttentionFusion` |
| **Objective / Loss** | Imbalanced probability calibration | `Poly1FocalLoss` & `FocalLoss` |
| **Tabular Ensemble** | High-bias small cohort regularization | `ClinicalEnsemble` with XGBoost / LightGBM |
| **Unsupervised MEM** | Pretraining on 76k unlabeled lab events | `MaskedTrajectoryPretrainer` (BERT-style 15% masking) |
| **RL Decision Policy** | Asymmetric clinical risk optimization | `ClinicalPolicyGradient` + `ClinicalRewardFunction` |
| **Semantic Explainer** | Natural language synthesis & verification | `ClinicalExplainer`, `SemanticTransformerDecoder`, `ExplanationValidator` |

---

## 3. Autonomous Antigravity Agent Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ANTIGRAVITY AGENT LOOP                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: HYPOTHESIS & EXPERIMENT PLANNING                              │
│ - Agent inspects log.txt, fold metrics, and confusion matrices.        │
│ - Hypothesizes specific parameter/feature adjustments.                 │
│ - Preserves group patient stratification (0% leakage).                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: PARALLEL VERIFICATION & BENCHMARKING                          │
│ - Runs smoke validation or full GPU cross-validation.                  │
│ - Collects AUROC, AUPRC, Brier Score, and Explanation Faithfulness.   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: SCIENTIFIC AUDIT & REWARD VERIFICATION                        │
│ - Executes tests/test_leakage_audit.py and ExplanationValidator.       │
│ - Flags unsupported clinical diagnoses or directional discrepancies.   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: PERSISTENCE & EVOLUTION                                       │
│ - Appends structured results to log.txt.                               │
│ - Updates best checkpoint weights and commits cleanly.                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Peak Performance Optimization Plan (Dataset-Constrained)

Because the cohort is small (100 patients, 129 admissions, 40 deaths), peak performance cannot be achieved by brute-force scaling. The agent executes optimization through four constrained dimensions:

### Dimension A: Unsupervised Pretraining Maximization
- **Mechanism**: Exploit all 76,074 rows in `data/real/labevents.csv`.
- **Action**: Run `MaskedTrajectoryPretrainer` with dynamic masking probabilities (15% to 20%) over 10 epochs.
- **Goal**: Enable the sequence transformer to learn physiological lab co-occurrences (e.g. Creatinine-BUN correlations) prior to seeing mortality labels.

### Dimension B: Multi-Modal Fusion Tuning
- **Mechanism**: Optimize `BidirectionalCrossAttentionFusion` temperature scaling and layer dropout.
- **Action**: Test fusion projection dimensions (d_model = 128 vs 256) and attention heads (4 vs 8).
- **Goal**: Align ClinicalBERT discharge note embeddings directly with physiological trajectory inflections.

### Dimension C: Calibration & Ensemble Regularization
- **Mechanism**: Hyperparameter tuning of `Poly1FocalLoss` (`epsilon` = 1.0, `gamma` = 2.0) and GBDT blend weights (`transformer_weight` = 0.55, `tree_weight` = 0.45).
- **Goal**: Resolve probability compression (breaking out of the 0.50 risk dead-zone) to drive Brier score below 0.20 and AUROC above 0.78.

### Dimension D: Reinforcement Learning Clinical Alignment
- **Mechanism**: Policy gradient fine-tuning using `ClinicalRewardFunction`.
- **Penalties**: False Negative (missed death) = -2.5, False Positive = -0.5, Unsupported Diagnosis = -1.5.
- **Goal**: Shift the classification threshold to maximize clinical utility and eliminate ungrounded clinical claims.

---

## 5. Modern AI Interface: Natural Language Input & Output

Modern AI interactions require decoupling clinicians from raw numeric tensors. The pipeline exposes natural language interfaces on both ends:

### 1. Natural Language Input Parsing
Clinicians or researchers query the model in free-text:
- *Query Example*: `"Analyze admission for patient with emergency heart failure, acute creatinine rise, and lactate > 4.0."`
- The system parses clinical entities using `ClinicalDictionary`, maps them to corresponding token representations (`AGE_*`, `GENDER_*`, `ADM_*`, `LAB_*_Q*`), and feeds them to the multimodal transformer.

### 2. Natural Language Output & Explainability
The model generates fluent, uncertainty-aware, clinically grounded text via `ClinicalExplainer`:
- **Risk Stratification**: Exact probability + uncertainty bounds (e.g., `"Elevated Risk of In-Hospital Mortality: 68.4% [95% CI: 62.1% - 74.7%]"`).
- **Physiological Trajectory Drivers**: Identifies top quantile deviations (e.g., `"Key findings: Elevated BUN (Q9), high blood lactate (Q8), and rapid downward trend in platelet count over 48 hours."`).
- **Explanation Audit**: Every narrative is vetted by `ExplanationValidator` to ensure 0% unsupported diagnostic claims.

---

## 6. Scientific & Technical Guardrails (Inviolable Rules)

1. **Zero Patient Overlap**: All splits must be partitioned using `StratifiedGroupKFold` or `GroupShuffleSplit` on `subject_id`.
2. **Zero Quantile Leakage**: Dynamic quantiles (`fit_quantiles`) must be fit strictly on training folds and transformed on validation/test folds.
3. **No Math/LaTeX Markup**: All metrics, reports, and logs must use clean Markdown plain-text formatting (e.g. `AUROC = 0.7556`, not math delimiters).
4. **No Arbitrary Variable Refactoring**: Keep established classes (`MultiModalMedicalTransformer`, `ClinicalExplainer`, `Time2Vec`, `ClinicalEnsemble`, `Poly1FocalLoss`).

---

## 7. Execution Commands for the Agent / GPU Runner

### Full 3-Stage GPU Benchmark
```bash
python cross_validate.py --multimodal --unified-3stage --unsupervised-epochs 5 --epochs 15 --rl-epochs 3 --label gpu-peak-3stage-benchmark
```

### High-Sensitivity Calibration Benchmark
```bash
python cross_validate.py --multimodal --loss poly_focal --epochs 20 --patience 5 --label gpu-poly-focal-tuned
```

### Pure Neural Cross-Attention (Ablation without GBDT)
```bash
python cross_validate.py --multimodal --no-ensemble --epochs 15 --label gpu-pure-neural-ablation
```

### Integrity & Leakage Verification Suite
```bash
python -m unittest discover tests
