"""
Loss functions for calibrated and imbalanced medical classification.
Includes Focal Loss and PolyLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard example mining in medical prediction.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.6, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Poly1FocalLoss(nn.Module):
    """
    Poly-1 formulation of Focal Loss (Leng et al., ICLR 2022).
    Adds a leading polynomial coefficient (epsilon_1 * (1 - p_t)^(gamma+1))
    to improve probability calibration and Brier score.
    """

    def __init__(self, alpha: float = 0.6, gamma: float = 2.0, epsilon: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        poly_term = self.epsilon * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma + 1.0)
        focal_loss = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma) * bce_loss
        loss = focal_loss + poly_term

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
