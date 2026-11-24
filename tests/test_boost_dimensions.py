"""
Unit tests for the 4 optimization dimensions:
- Dimension A: Unsupervised Pretraining Maximization & Dynamic Masking
- Dimension B: Multimodal Fusion Temperature Scaling
- Dimension C: Calibration & GBDT Ensembling
- Dimension D: RL Reward Rebalancing
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from medical_llm.models.unsupervised_pretrainer import MaskedTrajectoryDataset, MaskedTrajectoryPretrainer
from medical_llm.models.multimodal_transformer import BidirectionalCrossAttentionFusion, MultiModalMedicalTransformer
from medical_llm.models.ensemble import ClinicalEnsemble
from medical_llm.models.rl_policy import ClinicalRewardFunction, ClinicalPolicyGradient
from medical_llm.models.losses import Poly1FocalLoss
from medical_llm.models.tabular_features import extract_tabular_features
import pandas as pd


class TestBoostOptimizationDimensions(unittest.TestCase):

    def test_dimension_a_dynamic_masking_pretrainer(self):
        """Tests Dimension A dynamic masking probability range and pretrainer step."""
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "LAB_1": 5, "LAB_2": 6, "LAB_3": 7}
        seqs = [["LAB_1", "LAB_2", "LAB_3", "LAB_1", "LAB_2"]]

        dataset = MaskedTrajectoryDataset(seqs, vocab, mask_prob=(0.15, 0.20), max_len=8)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, (8,))

        pretrainer = MaskedTrajectoryPretrainer(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, max_len=8)
        logits, loss = pretrainer(sample["input_ids"].unsqueeze(0), sample["attention_mask"].unsqueeze(0), targets=sample["targets"].unsqueeze(0))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

    def test_dimension_b_multimodal_fusion_temperature(self):
        """Tests Dimension B BidirectionalCrossAttentionFusion with temperature scaling."""
        fusion = BidirectionalCrossAttentionFusion(d_model=32, nhead=4, dropout=0.1, temperature=0.8)
        h_labs = torch.randn(2, 10, 32)
        h_text = torch.randn(2, 8, 32)
        lab_mask = torch.ones(2, 10, dtype=torch.long)
        text_mask = torch.ones(2, 8, dtype=torch.long)

        out = fusion(h_labs, h_text, lab_mask=lab_mask, text_mask=text_mask)
        self.assertEqual(out.shape, (2, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_dimension_c_ensemble_gbdt_fallback_and_blending(self):
        """Tests Dimension C GBDT fitting, predict_proba, and blend weights."""
        ensemble = ClinicalEnsemble(transformer_weight=0.55, tree_weight=0.45)
        X = np.random.randn(20, 6)
        y = np.array([0] * 15 + [1] * 5)
        ensemble.fit_tree(X, y)

        transformer_probs = np.full(20, 0.5)
        blended = ensemble.predict_proba(transformer_probs, X)
        self.assertEqual(len(blended), 20)
        self.assertTrue(np.all(blended >= 0.0) and np.all(blended <= 1.0))

    def test_dimension_c_tabular_feature_alignment(self):
        """Tests that train and test tabular features stay column-aligned even with disjoint itemids."""
        train_adm = pd.DataFrame({"hadm_id": [1], "age": [60], "gender": ["M"], "admission_type": ["EMERGENCY"], "target": [1]})
        test_adm = pd.DataFrame({"hadm_id": [2], "age": [70], "gender": ["F"], "admission_type": ["ELECTIVE"], "target": [0]})
        train_labs = pd.DataFrame({"hadm_id": [1, 1], "itemid": [50971, 50983], "valuenum": [4.5, 140.0]})
        # Test labs has a completely different lab item (e.g. 50912) and only one of train's items
        test_labs = pd.DataFrame({"hadm_id": [2, 2], "itemid": [50912, 50971], "valuenum": [1.2, 4.0]})

        X_train, y_train, names_train = extract_tabular_features(train_adm, train_labs)
        train_itemids = [int(fn.split("_")[1]) for fn in names_train if fn.startswith("lab_") and fn.endswith("_min")]
        X_test, y_test, names_test = extract_tabular_features(test_adm, test_labs, itemids=train_itemids)

        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(names_train, names_test)

    def test_dimension_d_rl_reward_rebalancing(self):
        """Tests Dimension D rebalanced clinical utility rewards."""
        reward_fn = ClinicalRewardFunction()
        # Ensure rebalanced parameters penalize false positives more than default -0.5
        self.assertLess(reward_fn.fp_penalty, -0.8)
        # Ensure severe penalty on false negatives remains
        self.assertLessEqual(reward_fn.fn_penalty, -1.8)
        # Verify scores
        self.assertEqual(reward_fn.compute_decision_reward(1, 0), reward_fn.fp_penalty)
        self.assertEqual(reward_fn.compute_decision_reward(0, 1), reward_fn.fn_penalty)


if __name__ == "__main__":
    unittest.main()

