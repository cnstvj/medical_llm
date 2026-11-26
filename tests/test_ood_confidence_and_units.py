"""
Unit and integration tests for:
1. Unit normalization, conversion, and unconvertible mismatch warnings.
2. Prediction CONFIDENCE flag (HIGH / MEDIUM / LOW) and OOD pattern handling.
3. Recent-tail truncation preserving demographics + delta_times in datasets.
4. Scientific validity fixes (no note length leakage in tabular_features, fixed-threshold evaluator).
"""

import unittest
import warnings
import tempfile
import os
import numpy as np
import pandas as pd
import torch

from medical_llm.utils.units import (
    canonicalize_unit,
    fit_unit_registry,
    convert_unit_value,
    normalize_lab_units,
)
from medical_llm.utils.confidence import (
    assess_prediction_confidence,
    predict_with_confidence,
    predict_from_raw_labs,
)
from medical_llm.utils.quantiles import fit_quantiles, apply_quantiles_to_labevents
from medical_llm.data.dataset import MedicalDataset, MultiModalMedicalDataset, build_vocab
from medical_llm.models.transformer import MedicalTransformer
from medical_llm.models.clinical_explainer import ClinicalExplainer
from medical_llm.models.tabular_features import extract_tabular_features
from medical_llm.pipeline.evaluator import compute_metrics


class TestUnitConversionAndWarnings(unittest.TestCase):

    def test_unit_alias_and_general_conversion(self):
        self.assertEqual(canonicalize_unit("mg/dl"), "mg/dL")
        self.assertEqual(canonicalize_unit(" meq/l "), "mEq/L")
        self.assertEqual(canonicalize_unit("SECONDS"), "sec")
        self.assertEqual(canonicalize_unit("MOSM/KG"), "mOsm/kg")
        self.assertEqual(canonicalize_unit("uU/ML"), "uIU/mL")

        # 14.0 g/dL -> 140.0 g/L
        self.assertAlmostEqual(convert_unit_value(14.0, "g/dL", "g/L"), 140.0, places=4)
        # 5.5 mmol/L Glucose -> ~99.0858 mg/dL
        glu_mgdl = convert_unit_value(5.5, "mmol/L", "mg/dL", label="Glucose")
        self.assertIsNotNone(glu_mgdl)
        self.assertAlmostEqual(glu_mgdl, 5.5 * 18.0156, places=2)

    def test_normalize_lab_units_converts_and_warns_on_incompatible(self):
        train_labs = pd.DataFrame({
            "hadm_id": [1, 1, 1],
            "itemid": [50809, 50912, 51222],
            "label": ["Glucose", "Creatinine", "Hemoglobin"],
            "valuenum": [100.0, 1.0, 13.5],
            "valueuom": ["mg/dL", "mg/dL", "g/dL"],
            "flag": ["NORMAL", "NORMAL", "NORMAL"],
        })
        registry = fit_unit_registry(train_labs)
        self.assertEqual(registry[50809]["unit"], "mg/dL")
        self.assertEqual(registry[51222]["unit"], "g/dL")

        # Inference data:
        # - 50809 in mmol/L (convertible -> mg/dL)
        # - 51222 in g/L (convertible -> g/dL)
        # - 50912 in "widgets" (incompatible -> must warn and mask valuenum to NaN)
        # - 99999 unseen itemid
        test_labs = pd.DataFrame({
            "hadm_id": [2, 2, 2, 2],
            "itemid": [50809, 51222, 50912, 99999],
            "label": ["Glucose", "Hemoglobin", "Creatinine", "Novel Biomarker"],
            "valuenum": [5.5, 135.0, 88.4, 42.0],
            "valueuom": ["mmol/L", "g/L", "widgets", "U/mL"],
            "flag": ["NORMAL", "NORMAL", "ABNORMAL", "ABNORMAL"],
        })

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            norm_df, report = normalize_lab_units(test_labs, registry, emit_warnings=True)

        self.assertEqual(report.converted_count, 2)
        self.assertEqual(report.unconverted_count, 1)
        self.assertEqual(report.unseen_itemid_count, 1)
        self.assertEqual(report.unseen_itemids, [99999])
        self.assertTrue(len(caught) >= 1)
        self.assertIn("widgets", str(caught[0].message))

        # Converted values match expected training units
        self.assertAlmostEqual(norm_df.loc[0, "valuenum"], 5.5 * 18.0156, places=2)
        self.assertEqual(norm_df.loc[0, "valueuom"], "mg/dL")
        self.assertAlmostEqual(norm_df.loc[1, "valuenum"], 13.5, places=4)
        self.assertEqual(norm_df.loc[1, "valueuom"], "g/dL")

        # Incompatible unit value masked to NaN so it does not get bucketed in wrong scale
        self.assertTrue(pd.isna(norm_df.loc[2, "valuenum"]))

        # When applied to quantiles, masked row falls back to LAB_50912_ABNORMAL instead of bogus quantile
        qmap = fit_quantiles(train_labs, min_samples=1)
        proc = apply_quantiles_to_labevents(norm_df, qmap)
        self.assertEqual(proc.loc[2, "event_token"], "LAB_50912_ABNORMAL")


class TestConfidenceAndOODHandling(unittest.TestCase):

    def setUp(self):
        self.train_seqs = [
            ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_50809_Q2", "TIME_1TO6H", "LAB_50912_Q3", "TIME_1TO6H", "LAB_51222_Q1"],
            ["AGE_70", "GENDER_F", "ADM_ELECTIVE", "[SEP]", "LAB_50809_Q5", "TIME_6TO24H", "LAB_50912_Q4", "TIME_1TO6H", "LAB_51222_Q2"],
        ]
        self.vocab = build_vocab(self.train_seqs)

    def test_confidence_high_for_known_patterns(self):
        seq = ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_50809_Q2", "TIME_1TO6H", "LAB_50912_Q3", "LAB_51222_Q1"]
        res = assess_prediction_confidence(
            lab_tokens=seq,
            vocab=self.vocab,
            clinical_text="60-year-old male admitted via emergency with sepsis.",
            probability=0.82,
            threshold=0.50,
        )
        self.assertEqual(res["confidence"], "HIGH")
        self.assertGreaterEqual(res["confidence_score"], 0.85)

    def test_confidence_medium_for_partial_unforeseen(self):
        # 1 unseen lab out of 4 labs (75% known -> MEDIUM)
        seq = ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_50809_Q2", "LAB_50912_Q3", "LAB_51222_Q1", "LAB_99999_ABNORMAL"]
        res = assess_prediction_confidence(
            lab_tokens=seq,
            vocab=self.vocab,
            clinical_text="60-year-old male admitted via emergency with sepsis.",
            probability=0.80,
            threshold=0.50,
        )
        self.assertEqual(res["confidence"], "MEDIUM")

    def test_confidence_low_for_heavy_ood_or_empty(self):
        # Mostly unseen tokens -> LOW
        seq = ["AGE_90", "GENDER_X", "ADM_UNKNOWN", "[SEP]", "LAB_99991_Q1", "LAB_99992_Q2", "LAB_99993_Q3", "LAB_99994_Q4"]
        res = assess_prediction_confidence(
            lab_tokens=seq,
            vocab=self.vocab,
            clinical_text="Novel presentation.",
            probability=0.90,
            threshold=0.50,
        )
        self.assertEqual(res["confidence"], "LOW")

        # Empty input -> LOW
        res_empty = assess_prediction_confidence(lab_tokens=[], vocab=self.vocab, clinical_text="")
        self.assertEqual(res_empty["confidence"], "LOW")

    def test_predict_with_confidence_and_explainer(self):
        model = MedicalTransformer(vocab_size=len(self.vocab), d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_len=32)
        known_seq = ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_50809_Q2", "TIME_1TO6H", "LAB_50912_Q3", "LAB_51222_Q1"]
        ood_seq = ["LAB_99991_ABNORMAL", "LAB_99992_ABNORMAL", "LAB_99993_ABNORMAL"]

        preds = predict_with_confidence(
            model=model,
            lab_sequences=[known_seq, ood_seq],
            clinical_texts=["Known patient note.", ""],
            vocab=self.vocab,
            max_lab_len=32,
            threshold=0.5,
        )
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0]["confidence"], "HIGH")
        self.assertEqual(preds[1]["confidence"], "LOW")

        expl = ClinicalExplainer.generate_patient_rationale(
            probability=preds[1]["probability"],
            lab_tokens=ood_seq,
            clinical_text="",
            vocab=self.vocab,
        )
        self.assertEqual(expl["confidence"], "LOW")
        self.assertIn("confidence_score", expl)
        self.assertIn("confidence_reasons", expl)

    def test_dataset_recent_tail_truncation_and_backoff(self):
        vocab = build_vocab([["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_1_Q1"]])
        # Sequence longer than max_lab_len - 1 (9)
        long_seq = ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]"] + [f"LAB_{i}_ABNORMAL" for i in range(10)]
        long_deltas = [0.0, 0.0, 0.0, 0.0] + [float(i) for i in range(10)]

        ds = MultiModalMedicalDataset(
            [long_seq], ["Note"], [1], vocab=vocab, max_lab_len=10, delta_times=[long_deltas], unk_dropout=0.0
        )
        sample = ds[0]
        # Length 10: [CLS] + 4 demographic/SEP + 5 most recent labs (deltas 5, 6, 7, 8, 9)
        self.assertEqual(sample["lab_ids"][0].item(), vocab["[CLS]"])
        self.assertEqual(sample["lab_ids"][1].item(), vocab["AGE_60"])
        self.assertEqual(sample["lab_ids"][4].item(), vocab["[SEP]"])
        # Unseen LAB_i_ABNORMAL should map to LAB_UNKNOWN_ABNORMAL
        self.assertEqual(sample["lab_ids"][5].item(), vocab["LAB_UNKNOWN_ABNORMAL"])
        # Most recent delta_times preserved (5.0 .. 9.0)
        self.assertEqual(sample["delta_times"][5:].tolist(), [5.0, 6.0, 7.0, 8.0, 9.0])


class TestScientificValidityFixes(unittest.TestCase):

    def test_tabular_features_excludes_note_length_leakage(self):
        adm = pd.DataFrame({
            "hadm_id": [1, 2],
            "age": [60, 70],
            "gender": ["M", "F"],
            "admission_type": ["EMERGENCY", "ELECTIVE"],
            "diagnosis": ["SEPSIS", "CHF"],
            "target": [1, 0],
        })
        labs = pd.DataFrame({"hadm_id": [1, 2], "itemid": [50800, 50800], "valuenum": [10.0, 20.0]})
        reports = pd.DataFrame({"hadm_id": [1], "clean_text": ["severe clinical deterioration"]})
        _, _, feature_names = extract_tabular_features(adm, labs, reports)
        self.assertNotIn("note_char_count", feature_names)
        self.assertNotIn("note_word_count", feature_names)

    def test_compute_metrics_respects_fixed_threshold(self):
        probs = [0.1, 0.4, 0.6, 0.9]
        labels = [0, 0, 1, 1]
        m = compute_metrics(probs, labels, threshold=0.55)
        self.assertAlmostEqual(m["threshold"], 0.55, places=4)
        self.assertEqual(m["f1"], 1.0)

    def test_two_step_unit_conversion(self):
        # Hemoglobin: 135 g/L -> 13.5 g/dL -> 13.5 * 0.6206 mmol/L
        hb_mmol = convert_unit_value(135.0, "g/L", "mmol/L", label="Hemoglobin")
        self.assertIsNotNone(hb_mmol)
        self.assertAlmostEqual(hb_mmol, 13.5 * 0.6206, places=3)

        # Glucose: 1000 mg/L -> 100 mg/dL -> 100 / 18.0156 mmol/L
        glu_mmol = convert_unit_value(1000.0, "mg/L", "mmol/L", label="Glucose")
        self.assertIsNotNone(glu_mmol)
        self.assertAlmostEqual(glu_mmol, 100.0 / 18.0156, places=3)

    def test_near_threshold_score_consistency(self):
        vocab = build_vocab([["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_50809_Q2", "LAB_50912_Q3", "LAB_51222_Q1"]])
        seq = ["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]", "LAB_50809_Q2", "LAB_50912_Q3", "LAB_51222_Q1"]
        res = assess_prediction_confidence(
            lab_tokens=seq,
            vocab=vocab,
            clinical_text="60-year-old male admitted via emergency.",
            probability=0.51,
            threshold=0.50,
        )
        self.assertEqual(res["confidence"], "HIGH")
        self.assertLess(res["confidence_score"], 1.0)

    def test_predict_from_raw_labs_end_to_end(self):
        train_labs = pd.DataFrame({
            "hadm_id": [1, 1, 1],
            "itemid": [50809, 50912, 51222],
            "label": ["Glucose", "Creatinine", "Hemoglobin"],
            "charttime": ["2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00"],
            "valuenum": [100.0, 1.0, 13.5],
            "valueuom": ["mg/dL", "mg/dL", "g/dL"],
            "flag": ["NORMAL", "NORMAL", "NORMAL"],
        })
        registry = fit_unit_registry(train_labs)
        qmap = fit_quantiles(train_labs, min_samples=1)
        train_proc = apply_quantiles_to_labevents(train_labs, qmap)
        train_proc["charttime"] = pd.to_datetime(train_proc["charttime"])
        from medical_llm.data.sequence_builder import build_temporal_sequences
        train_grouped = build_temporal_sequences(train_proc, experiment_mode="FULL")
        train_seqs = [["AGE_60", "GENDER_M", "ADM_EMERGENCY", "[SEP]"] + train_grouped["event_token"].iloc[0]]
        vocab = build_vocab(train_seqs)
        model = MedicalTransformer(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_len=32)

        # Admission 2: convertible Glucose mmol/L -> mg/dL, Admission 3: unconvertible "widgets" -> warns & MEDIUM/LOW
        new_adm = pd.DataFrame({
            "hadm_id": [2, 3],
            "age": [60, 60],
            "gender": ["M", "M"],
            "admission_type": ["EMERGENCY", "EMERGENCY"],
            "diagnosis": ["SEPSIS", "SEPSIS"],
        })
        new_labs = pd.DataFrame({
            "hadm_id": [2, 2, 2, 3],
            "itemid": [50809, 50912, 51222, 50809],
            "label": ["Glucose", "Creatinine", "Hemoglobin", "Glucose"],
            "charttime": ["2020-01-02 01:00:00", "2020-01-02 02:00:00", "2020-01-02 03:00:00", "2020-01-03 01:00:00"],
            "valuenum": [5.5, 1.0, 135.0, 99.0],
            "valueuom": ["mmol/L", "mg/dL", "g/L", "widgets"],
            "flag": ["NORMAL", "NORMAL", "NORMAL", "ABNORMAL"],
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preds = predict_from_raw_labs(
                model=model,
                admissions_df=new_adm,
                labevents_df=new_labs,
                vocab=vocab,
                quantile_map=qmap,
                unit_registry=registry,
                max_lab_len=32,
                emit_warnings=True,
            )
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0]["confidence"], "HIGH")
        self.assertIn(preds[1]["confidence"], ("MEDIUM", "LOW"))
        self.assertTrue(len(caught) >= 1)
        self.assertTrue(len(preds[1]["warnings"]) >= 1)


if __name__ == "__main__":
    unittest.main()
