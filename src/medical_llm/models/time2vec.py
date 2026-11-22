"""
Time2Vec and Continuous-Time Fourier Embedding for Irregular Clinical Time-Series.
Transforms irregular continuous delta-times into learned periodic and linear frequency features.
"""

import math
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time (Kazemi et al., 2019).
    Outputs:
        t2v[0] = w0 * t + phi0  (linear trend component)
        t2v[i] = sin(wi * t + phii) for i = 1..k-1 (periodic harmonic frequencies)
    """

    def __init__(self, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim
        # Linear component
        self.w0 = nn.Parameter(torch.empty(1, 1))
        self.phi0 = nn.Parameter(torch.empty(1, 1))
        # Periodic components
        self.w = nn.Parameter(torch.empty(1, time_dim - 1))
        self.phi = nn.Parameter(torch.empty(1, time_dim - 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.w0, -0.1, 0.1)
        nn.init.uniform_(self.phi0, -0.1, 0.1)
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.phi, -0.1, 0.1)

    def forward(self, delta_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_time: Tensor of shape (batch_size, seq_len) representing delta hours.
        Returns:
            Tensor of shape (batch_size, seq_len, time_dim)
        """
        # (B, L, 1)
        tau = delta_time.unsqueeze(-1).float()
        linear = self.w0 * tau + self.phi0  # (B, L, 1)
        periodic = torch.sin(tau * self.w + self.phi)  # (B, L, time_dim - 1)
        return torch.cat([linear, periodic], dim=-1)
