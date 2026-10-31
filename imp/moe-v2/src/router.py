"""
Self-Contained Spectral-Token Router and Soft Gating Mechanisms for Single-Backbone MoE.

References:
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts",
      ICLR 2024. arXiv:2308.00951.
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts",
      NeurIPS 2021. arXiv:2106.05974.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization
      in Mixture-of-Experts Language Models", ACL 2024. arXiv:2401.06066.
    - FiLM / Conditioned Transformers: Perez et al., "FiLM: Visual Reasoning with
      a General Conditioning Layer", AAAI 2018. arXiv:1709.07871.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfContainedSpectralTokenRouter(nn.Module):
    """
    Self-Contained Dynamic Routing Module that estimates expert gating weights
    purely from the input noise signal n_hat and intermediate transformer representations,
    eliminating any dependency on Oracle SNR or clean speech references at inference.

    Input features:
        - 6 physical/spectral features from AcousticFeatureExtractor (log-RMS, crest factor,
          frame kurtosis, spectral flatness, low/speech energy ratio, high/speech energy ratio)
        - 192-dim intermediate transformer token representation (pooled at Block 9)

    Output:
        - alpha: Continuous soft gating weight in [0, 1]
          (alpha -> 0: Low-SNR regime, alpha -> 1: High-SNR regime)
        - routing_logits: Unnormalized logits for 2 routed experts
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

        # Linear projection for intermediate transformer token
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )

        in_total = spectral_dim + token_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_total, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [Logit_Low, Logit_High]
        )

    def forward(
        self,
        spectral_feat: Tensor,
        intermediate_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            spectral_feat: [B, 6] from AcousticFeatureExtractor
            intermediate_token: [B, 192] from Block 9 of CED backbone
        Returns:
            alpha: [B, 1] gating weight for High-Expert (1 - alpha for Low-Expert)
            gates: [B, 2] normalized softmax probabilities over [Low, High]
        """
        # Detach or keep gradients based on architectural coupling
        token_h = self.token_proj(intermediate_token)
        fused_feat = torch.cat([spectral_feat, token_h], dim=-1)
        logits = self.mlp(fused_feat)  # [B, 2]

        gates = F.softmax(logits, dim=-1)  # [B, 2]
        alpha = gates[:, 1:2]  # High-Expert weight
        return alpha, gates

    def compute_target_soft_label(self, snr_db: Tensor) -> Tensor:
        """
        Compute continuous ground truth routing label from SNR:
        y_route = Sigmoid((SNR - 15.0) / 2.5)

        Example values:
            SNR = -5 dB -> y_route = 0.0003
            SNR =  0 dB -> y_route = 0.0025
            SNR =  5 dB -> y_route = 0.0180
            SNR = 10 dB -> y_route = 0.1192
            SNR = 15 dB -> y_route = 0.5000 (Smooth transition zone)
            SNR = 20 dB -> y_route = 0.8808 (High-SNR speech leakage regime)
        """
        z = (snr_db.float() - self.transition_center_db) / self.transition_tau
        return torch.sigmoid(z).unsqueeze(-1)

    def compute_router_loss(self, alpha: Tensor, snr_db: Tensor) -> Tensor:
        """
        Supervised Binary Cross Entropy loss for the routing gate during training.
        """
        target = self.compute_target_soft_label(snr_db)
        return F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), target.float())
