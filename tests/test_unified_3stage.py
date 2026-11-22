"""
Unit tests for Unified 3-Stage Learning Pipeline:
- Unsupervised Masked Trajectory Modeling (Stage 1)
- Supervised Fine-Tuning (Stage 2)
- Reinforcement Learning Policy Gradient with Clinical Reward (Stage 3)
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medical_llm.models.unsupervised_pretrainer import MaskedTrajectoryDataset, MaskedTrajectoryPretrainer
from medical_llm.models.rl_policy import ClinicalRewardFunction, ClinicalPolicyGradient
from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer


class TestUnifiedThreeStagePipeline(unittest.TestCase):

    def test_unsupervised_masked_dataset_and_pretrainer(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "LAB_50800_Q1": 5, "LAB_50800_Q2": 6}
        seqs = [["LAB_50800_Q1", "LAB_50800_Q2"], ["LAB_50800_Q2", "LAB_50800_Q1"]]

        dataset = MaskedTrajectoryDataset(seqs, vocab, mask_prob=0.5, max_len=8)
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, (8,))
        self.assertEqual(sample["targets"].shape, (8,))

        pretrainer = MaskedTrajectoryPretrainer(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, max_len=8)
        input_ids = sample["input_ids"].unsqueeze(0)
        mask = sample["attention_mask"].unsqueeze(0)
        targets = sample["targets"].unsqueeze(0)

        logits, loss = pretrainer(input_ids, mask, targets=targets)
        self.assertEqual(logits.shape, (1, 8, len(vocab)))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

    def test_clinical_reward_function(self):
        reward_fn = ClinicalRewardFunction(tp_reward=1.0, tn_reward=0.5, fp_penalty=-0.5, fn_penalty=-2.5)
        # TP
        self.assertEqual(reward_fn.compute_decision_reward(1, 1), 1.0)
        # TN
        self.assertEqual(reward_fn.compute_decision_reward(0, 0), 0.5)
        # FP
        self.assertEqual(reward_fn.compute_decision_reward(1, 0), -0.5)
        # FN (asymmetric severe penalty)
        self.assertEqual(reward_fn.compute_decision_reward(0, 1), -2.5)

    def test_rl_policy_gradient_step(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "A": 5}
        model = MultiModalMedicalTransformer(vocab_size=len(vocab), d_model=16, nhead=2, num_layers=1, max_lab_len=8)
        policy = ClinicalPolicyGradient(backbone_model=model)

        lab_ids = torch.randint(0, 6, (2, 8))
        lab_mask = torch.ones(2, 8, dtype=torch.long)
        text_ids = torch.zeros(2, 16, dtype=torch.long)
        text_mask = torch.zeros(2, 16, dtype=torch.long)
        targets = torch.tensor([1, 0])

        probs = policy(lab_ids, lab_mask, text_ids, text_mask)
        self.assertEqual(probs.shape, (2, 2))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5))

        actions, log_probs = policy.sample_action(probs)
        loss = policy.compute_policy_loss(log_probs, actions, targets)
        self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
