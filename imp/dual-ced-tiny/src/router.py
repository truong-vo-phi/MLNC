"""Router modules, including the learned self-contained MoE-v2 routing path."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def estimate_snr_db(mixture: Tensor, noise: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Estimates Signal-to-Noise Ratio (SNR) in dB from mixture and noise:
        s_hat = mixture - noise
        P_signal = mean(s_hat^2)
        P_noise = mean(noise^2)
        SNR_dB = 10 * log10(P_signal / P_noise)
    """
    if mixture.ndim == 1:
        mixture = mixture.unsqueeze(0)
    if noise.ndim == 1:
        noise = noise.unsqueeze(0)

    clean = mixture - noise
    power_clean = torch.mean(clean ** 2, dim=-1) + eps
    power_noise = torch.mean(noise ** 2, dim=-1) + eps
    return 10.0 * torch.log10(power_clean / power_noise)


class RoutingDecision(NamedTuple):
    alpha: Tensor             # Blending factor for High-Expert in [0, 1] [B, 1]
    is_high_snr: Tensor       # Boolean flag indicating SNR >= threshold [B]
    estimated_snr_db: Tensor  # SNR in dB [B]


class EnergySNRRouter(nn.Module):
    """
    Energy and SNR-based Router for Dual-CED-Tiny:
    Routes input audio between Low-Expert (SNR < 18 dB) and High-Expert (SNR >= 18 dB).
    """

    def __init__(
        self,
        threshold_db: float = 18.0,
        temp_scale: float = 2.0,
        mode: str = "energy_threshold",
    ) -> None:
        super().__init__()
        self.threshold_db = threshold_db
        self.temp_scale = temp_scale
        self.mode = mode

    def estimate_energy_snr(self, wave: Tensor) -> Tensor:
        """
        Estimates energy envelope based heuristic SNR proxy if clean reference is unavailable:
        Compares high-energy frame power to noise-floor power.
        """
        # Split 4s audio into 100ms frames (1600 samples)
        B, T = wave.shape[0], wave.shape[1]
        frame_len = 1600
        num_frames = T // frame_len
        frames = wave[:, : num_frames * frame_len].reshape(B, num_frames, frame_len)
        frame_power = torch.mean(frames ** 2, dim=-1) + 1e-8

        # Signal power approx (top 20% loudest frames)
        k = max(1, int(num_frames * 0.2))
        top_power = torch.topk(frame_power, k=k, dim=-1)[0].mean(dim=-1)
        # Noise power approx (bottom 20% quietest frames)
        bot_power = torch.topk(-frame_power, k=k, dim=-1)[0].abs().mean(dim=-1)

        snr_db = 10.0 * torch.log10(top_power / (bot_power + 1e-8))
        return snr_db

    def forward(
        self,
        wave: Tensor,
        snrs_db: Tensor | None = None,
        mixture: Tensor | None = None,
    ) -> RoutingDecision:
        """
        Computes routing decision (alpha, is_high_snr, snr_db).
        """
        if snrs_db is not None:
            snr = snrs_db.to(wave.device)
        elif mixture is not None:
            snr = estimate_snr_db(mixture, wave)
        else:
            snr = self.estimate_energy_snr(wave)

        is_high = snr >= self.threshold_db

        if self.mode == "hard" or self.mode == "energy_threshold":
            alpha = is_high.float().unsqueeze(-1)  # 1.0 for high, 0.0 for low
        elif self.mode == "soft_blending":
            # Continuous sigmoid blending around threshold
            diff = (snr - self.threshold_db) / self.temp_scale
            alpha = torch.sigmoid(diff).unsqueeze(-1)
        else:
            alpha = is_high.float().unsqueeze(-1)

        return RoutingDecision(alpha=alpha, is_high_snr=is_high, estimated_snr_db=snr)


class SelfContainedSpectralTokenRouter(nn.Module):
    """Learn a soft Low/High gate without reading validation or inference SNR.

    The router combines six waveform morphology features with a projected
    192-dimensional intermediate CED token, following ``imp/moe-v2``.
    ``alpha`` is the High-expert probability; the returned ``gates`` are
    ordered ``[Low, High]``.
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
        if transition_tau <= 0:
            raise ValueError("transition_tau must be positive")
        self.transition_center_db = float(transition_center_db)
        self.transition_tau = float(transition_tau)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(spectral_dim + token_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, spectral_feat: Tensor, intermediate_token: Tensor) -> tuple[Tensor, Tensor]:
        if spectral_feat.ndim != 2 or intermediate_token.ndim != 2:
            raise ValueError("router inputs must be [B, features] and [B, token_dim]")
        if spectral_feat.shape[0] != intermediate_token.shape[0]:
            raise ValueError("router input batch sizes must match")
        features = torch.cat([spectral_feat.float(), self.token_proj(intermediate_token.float())], dim=-1)
        gates = F.softmax(self.gate(features), dim=-1)
        return gates[:, 1:2], gates

    def compute_target_soft_label(self, snr_db: Tensor) -> Tensor:
        """Convert train-only SNR metadata to the smooth MoE-v2 gate target."""
        scaled = (snr_db.float() - self.transition_center_db) / self.transition_tau
        return torch.sigmoid(scaled).unsqueeze(-1)

    def compute_router_loss(self, alpha: Tensor, snr_db: Tensor) -> Tensor:
        target = self.compute_target_soft_label(snr_db)
        return F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), target)
