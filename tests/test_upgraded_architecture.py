"""
Unit tests for upgraded model architecture components:
- Time2Vec continuous temporal embedding
- Poly1FocalLoss & FocalLoss
- Bidirectional Cross Attention Fusion
- Tabular feature extraction
- Clinical Ensemble (XGBoost / LightGBM)
"""

import unittest
import torch
import numpy as np
import pandas as pd

from medical_llm.models.time2vec import Time2Vec
from medical_llm.models.losses import FocalLoss, Poly1FocalLoss
from medical_llm.models.multimodal_transformer import BidirectionalCrossAttentionFusion, MultiModalMedicalTransformer
from medical_llm.models.tabular_features import extract_tabular_features
from medical_llm.models.ensemble import ClinicalEnsemble
from medical_llm.config import parse_pipeline_args, PipelineConfig


class TestUpgradedArchitecture(unittest.TestCase):

    def test_time2vec_shape_and_forward(self):
        t2v = Time2Vec(time_dim=16)
        delta_t = torch.tensor([[0.0, 1.5, 4.2], [0.0, 12.0, 48.5]])
        out = t2v(delta_t)
        self.assertEqual(out.shape, (2, 3, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_focal_loss(self):
        fl = FocalLoss(alpha=0.6, gamma=2.0)
        logits = torch.tensor([1.5, -2.0, 0.5])
        targets = torch.tensor([1.0, 0.0, 1.0])
        loss = fl(logits, targets)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))

    def test_poly1_focal_loss(self):
        poly_fl = Poly1FocalLoss(alpha=0.6, gamma=2.0, epsilon=1.0)
        logits = torch.tensor([2.0, -1.5, 0.0])
        targets = torch.tensor([1.0, 0.0, 0.0])
        loss = poly_fl(logits, targets)
        self.assertTrue(loss.item() > 0)
        self.assertFalse(torch.isnan(loss))

    def test_cross_attention_fusion(self):
        fusion = BidirectionalCrossAttentionFusion(d_model=32, nhead=2)
        h_labs = torch.randn(2, 10, 32)
        h_text = torch.randn(2, 8, 32)
        lab_mask = torch.ones(2, 10)
        text_mask = torch.ones(2, 8)
        out = fusion(h_labs, h_text, lab_mask=lab_mask, text_mask=text_mask)
        self.assertEqual(out.shape, (2, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_tabular_features_and_ensemble(self):
        adm_df = pd.DataFrame({
            "hadm_id": [101, 102],
            "age": [65, 45],
            "gender": ["M", "F"],
            "admission_type": ["EMERGENCY", "ELECTIVE"],
            "target": [1, 0]
        })
        labs_df = pd.DataFrame({
            "hadm_id": [101, 101, 102],
            "itemid": [50800, 50800, 50800],
            "valuenum": [12.5, 14.0, 10.0]
        })
        reports_df = pd.DataFrame({
            "hadm_id": [101, 102],
            "clean_text": ["patient with acute fever", "routine checkup"]
        })
        X, y, names = extract_tabular_features(adm_df, labs_df, reports_df)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(len(y), 2)

        ensemble = ClinicalEnsemble(transformer_weight=0.6, tree_weight=0.4, tree_type="xgboost")
        ensemble.fit_tree(X, y)
        trans_probs = np.array([0.7, 0.2])
        blended = ensemble.predict_proba(trans_probs, X)
        self.assertEqual(len(blended), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in blended))

    def test_pipeline_config_and_args_parsing(self):
        args = [
            "--multimodal",
            "--unified-3stage",
            "--unsupervised-epochs", "5",
            "--epochs", "15",
            "--rl-epochs", "3",
            "--label", "gpu-unified-3stage",
        ]
        cfg, dummy, label = parse_pipeline_args(args)
        self.assertTrue(cfg.use_multimodal_nlp)
        self.assertTrue(cfg.enable_unsupervised_pretraining)
        self.assertTrue(cfg.enable_rl_policy_tuning)
        self.assertEqual(cfg.unsupervised_epochs, 5)
        self.assertEqual(cfg.epochs, 15)
        self.assertEqual(cfg.rl_epochs, 3)
        self.assertEqual(label, "gpu-unified-3stage")
        self.assertEqual(cfg.text_learning_rate, 2e-5)


if __name__ == "__main__":
    unittest.main()
