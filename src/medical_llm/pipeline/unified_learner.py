"""
Unified 3-Stage Training Pipeline:
Stage 1: Unsupervised Self-Supervised Pretraining (Masked Trajectory Modeling)
Stage 2: Supervised Fine-Tuning (Multimodal Mortality Classification)
Stage 3: Reinforcement Learning (Policy Gradient with Asymmetric Clinical Reward)
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from medical_llm.config import PipelineConfig, DEFAULT_CONFIG
from medical_llm.models.unsupervised_pretrainer import MaskedTrajectoryDataset, MaskedTrajectoryPretrainer
from medical_llm.models.multimodal_transformer import MultiModalMedicalTransformer
from medical_llm.models.rl_policy import ClinicalPolicyGradient, ClinicalRewardFunction
from medical_llm.data.dataset import build_vocab, MultiModalMedicalDataset


def train_unsupervised_stage(
    raw_sequences: List[List[str]],
    vocab: Dict[str, int],
    config: PipelineConfig = DEFAULT_CONFIG,
    epochs: int = 5,
    device: Optional[torch.device] = None,
) -> MaskedTrajectoryPretrainer:
    """Stage 1: Self-supervised pretraining on unlabeled trajectories."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[STAGE 1: UNSUPERVISED PRETRAINING] Running {epochs} epochs of Masked Trajectory Modeling...", flush=True)
    dataset = MaskedTrajectoryDataset(raw_sequences, vocab, mask_prob=0.15, max_len=config.max_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    pretrainer = MaskedTrajectoryPretrainer(
        vocab_size=len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    optimizer = optim.AdamW(pretrainer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(epochs):
        pretrainer.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            _, loss = pretrainer(input_ids, mask, targets=targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Unsupervised Epoch {epoch+1}/{epochs} - Reconstruction Loss: {total_loss:.4f}", flush=True)

    return pretrainer


def train_rl_stage(
    model: MultiModalMedicalTransformer,
    train_loader: DataLoader,
    config: PipelineConfig = DEFAULT_CONFIG,
    epochs: int = 3,
    device: Optional[torch.device] = None,
) -> ClinicalPolicyGradient:
    """Stage 3: Reinforcement learning policy optimization using clinical utility reward."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[STAGE 3: REINFORCEMENT LEARNING] Fine-tuning policy gradient for {epochs} epochs...", flush=True)
    policy = ClinicalPolicyGradient(backbone_model=model).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=config.learning_rate * 0.1, weight_decay=config.weight_decay)

    for epoch in range(epochs):
        policy.train()
        total_loss = 0.0
        for batch in train_loader:
            lab_ids = batch["lab_ids"].to(device)
            lab_mask = batch["lab_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            delta_times = batch.get("delta_times", None)
            if delta_times is not None:
                delta_times = delta_times.to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            action_probs = policy(lab_ids, lab_mask, text_ids, text_mask, delta_times=delta_times)
            actions, log_probs = policy.sample_action(action_probs)
            rl_loss = policy.compute_policy_loss(log_probs, actions, y)

            rl_loss.backward()
            optimizer.step()
            total_loss += rl_loss.item()

        print(f"  RL Policy Epoch {epoch+1}/{epochs} - Policy Gradient Loss: {total_loss:.4f}", flush=True)

    return policy
