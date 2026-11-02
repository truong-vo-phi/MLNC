"""
Self-Contained Spectral-Token Router for Single-Backbone MoE (Zero-Oracle SNR).

Combines:
1. 6 Acoustic physical/spectral features (f_spec in R^6) directly from audio waveform.
2. Semantic contextual CLS token representation (h_9 in R^192) extracted after Transformer Block 9.

Outputs continuous soft gate weight alpha in (0, 1) for High-SNR Expert (and 1 - alpha for Low-SNR Expert).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfContainedSpectralTokenRouter(nn.Module):
    """
    Predicts continuous soft routing weight alpha in (0, 1):
    1. CLS token h_9 [B, 192] -> LayerNorm(192) -> Linear(192, 16) -> GELU -> [B, 16]
    2. Concat with 6 spectral features [B, 6] -> [B, 22]
    3. MLP: Linear(22, 32) -> LayerNorm(32) -> GELU -> Linear(32, 2) -> [r_low, r_high]
    4. Softmax: alpha = exp(r_high) / (exp(r_low) + exp(r_high))
    """
    def __init__(
        self,
        spectral_dim: int = 6,
        token_dim: int = 192,
        token_proj_dim: int = 16,
        hidden_dim: int = 32,
        transition_center_db: float = 15.0,
        transition_tau: float = 2.5,
    ) -> None:
        super().__init__()
        self.transition_center_db = transition_center_db
        self.transition_tau = transition_tau

        # Token projection branch (192 -> 16)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )

        # Multi-level cross-feature interaction MLP (22 -> 32 -> 2)
        in_total = spectral_dim + token_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_total, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [r_low, r_high]
        )

    def forward(
        self,
        spectral_feat: Tensor,
        intermediate_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            spectral_feat: [B, 6] from AcousticFeatureExtractor
            intermediate_token: [B, 192] CLS token after Block 9
        Returns:
            alpha: [B, 1] soft routing weight for High-SNR Expert
            gates: [B, 2] softmax probabilities [p_low, p_high]
        """
        token_h = self.token_proj(intermediate_token)  # [B, 16]
        fused_feat = torch.cat([spectral_feat, token_h], dim=-1)  # [B, 22]
        logits = self.mlp(fused_feat)  # [B, 2]
        gates = F.softmax(logits, dim=-1)  # [B, 2]
        alpha = gates[:, 1:2]  # [B, 1] High-SNR expert weight
        return alpha, gates

    def compute_target_soft_label(self, snr_db: Tensor) -> Tensor:
        """
        Continuous soft target from ground-truth SNR during training:
        y_target = Sigmoid((SNR - 15.0) / 2.5)
        """
        z = (snr_db.float() - self.transition_center_db) / self.transition_tau
        return torch.sigmoid(z).unsqueeze(-1)

    def compute_router_loss(self, alpha: Tensor, snr_db: Tensor) -> Tensor:
        """Binary cross-entropy loss with soft SNR target."""
        target = self.compute_target_soft_label(snr_db)
        return F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), target.float())

