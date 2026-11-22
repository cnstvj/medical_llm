"""
Reinforcement Learning Policy & Clinical Reward Optimization Module.
Implements Policy Gradient / REINFORCE for optimizing clinical decision utility,
penalizing asymmetric false negatives (missed mortality) and rewarding validated faithful explanations.
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from medical_llm.semantic.validator import ExplanationValidator


class ClinicalRewardFunction:
    """
    Computes scalar clinical utility rewards based on:
    1. Outcome utility matrix (heavy asymmetric penalty for false negatives)
    2. Explanation faithfulness and absence of hallucinated diagnoses / treatments
    """

    def __init__(
        self,
        tp_reward: float = 1.0,
        tn_reward: float = 0.5,
        fp_penalty: float = -0.5,
        fn_penalty: float = -2.5,  # Missed mortality is catastrophic in clinical practice
        hallucination_penalty: float = -1.5,
    ):
        self.tp_reward = tp_reward
        self.tn_reward = tn_reward
        self.fp_penalty = fp_penalty
        self.fn_penalty = fn_penalty
        self.hallucination_penalty = hallucination_penalty
        self.validator = ExplanationValidator()

    def compute_decision_reward(self, prediction: int, target: int) -> float:
        """Utility matrix score for clinical decisions."""
        if prediction == 1 and target == 1:
            return self.tp_reward
        elif prediction == 0 and target == 0:
            return self.tn_reward
        elif prediction == 1 and target == 0:
            return self.fp_penalty
        else:
            return self.fn_penalty

    def compute_explanation_reward(self, explanation_text: str, semantic_repr: Any) -> float:
        """Reward penalty based on explanation verification."""
        validation = self.validator.validate(explanation_text, semantic_repr)
        if validation["is_valid"]:
            return 0.5
        # Penalize per violation
        num_violations = len(validation.get("violations", []))
        return self.hallucination_penalty * max(num_violations, 1)

    def total_reward(
        self,
        prediction: int,
        target: int,
        explanation_text: Optional[str] = None,
        semantic_repr: Optional[Any] = None,
    ) -> float:
        reward = self.compute_decision_reward(prediction, target)
        if explanation_text is not None and semantic_repr is not None:
            reward += self.compute_explanation_reward(explanation_text, semantic_repr)
        return float(reward)


class ClinicalPolicyGradient(nn.Module):
    """
    Policy Network wrapper for optimizing clinical triage decisions via REINFORCE policy gradients.
    """

    def __init__(self, backbone_model: nn.Module):
        super().__init__()
        self.backbone = backbone_model
        self.reward_fn = ClinicalRewardFunction()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Returns action probabilities for [class_0 (discharge/low risk), class_1 (alert/high risk)]."""
        logits = self.backbone(*args, **kwargs)  # (batch_size,)
        # Reshape to 2-class policy distribution
        p_alert = torch.sigmoid(logits).unsqueeze(-1)
        p_safe = 1.0 - p_alert
        action_probs = torch.cat([p_safe, p_alert], dim=-1)
        # Normalize to ensure sum is exactly 1.0 across classes
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        return action_probs

    def sample_action(self, action_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples categorical decision actions and returns (action, log_prob).
        """
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def compute_policy_loss(
        self, log_probs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes REINFORCE policy gradient loss:
        L = - E [ log pi(a|s) * Reward(a, y) ]
        """
        rewards = []
        for a, y in zip(actions.cpu().numpy(), targets.cpu().numpy()):
            r = self.reward_fn.compute_decision_reward(int(a), int(y))
            rewards.append(r)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=log_probs.device)
        # Normalize rewards for gradient variance reduction
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-6)

        policy_loss = -(log_probs * rewards_tensor).mean()
        return policy_loss
