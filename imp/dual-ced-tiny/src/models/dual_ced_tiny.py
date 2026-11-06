"""Dual CED-Tiny experts with GREP, Alpha Post-LN, and learned acoustic-token routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseAudioClassifier
from src.models.ced_tiny import CEDTiny, load_ced_pretrained
from src.frontend import AcousticFeatureExtractor
from src.router import SelfContainedSpectralTokenRouter


class DualCEDTiny(BaseAudioClassifier):
    """
    Two full CED-Tiny backbones. The router blends their logits using waveform
    acoustic cues and intermediate CLS tokens; ground-truth SNR is never read
    by validation/inference routing.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 64,
        num_frames: int = 398,
        snr_threshold_db: float | None = None,
        router_mode: str | None = None,
        dropout: float = 0.0,
        pretrained: bool = False,
        pretrained_source: str | Path = "mispeech/ced-tiny",
        use_grep: bool = True,
        post_ln: bool = True,
        pre_ln: bool = True,
        router_transition_center_db: float = 15.0,
        router_transition_tau: float = 2.5,
        router_token_proj_dim: int = 16,
        router_hidden_dim: int = 32,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )

        # 1. Low-SNR Expert (GREP + Alpha Post-LN CED-Tiny)
        self.low_expert = CEDTiny(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
            dropout=dropout,
            use_grep=use_grep,
            post_ln=post_ln,
            pre_ln=pre_ln,
        )

        # 2. High-SNR Expert (GREP + Alpha Post-LN CED-Tiny)
        self.high_expert = CEDTiny(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
            dropout=dropout,
            use_grep=use_grep,
            post_ln=post_ln,
            pre_ln=pre_ln,
        )

        # 3. Self-contained spectral-token router; SNR metadata is train-target only.
        self.acoustic_extractor = AcousticFeatureExtractor()
        self.router = SelfContainedSpectralTokenRouter(
            spectral_dim=6,
            token_dim=192,
            token_proj_dim=router_token_proj_dim,
            hidden_dim=router_hidden_dim,
            transition_center_db=router_transition_center_db,
            transition_tau=router_transition_tau,
        )

        if pretrained:
            self.load_pretrained_backbones(pretrained_source)

    def load_pretrained_backbones(self, repo_or_path: str | Path = "mispeech/ced-tiny") -> tuple[int, int]:
        """Loads AudioSet-pretrained CED-Tiny weights into both Low and High experts."""
        n_low = load_ced_pretrained(self.low_expert, repo_or_path)
        n_high = load_ced_pretrained(self.high_expert, repo_or_path)
        return n_low, n_high

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Extracts latent features from both Low and High experts."""
        feat_low = self.low_expert.forward_features(x)
        feat_high = self.high_expert.forward_features(x)
        return feat_low, feat_high

    def forward_head(self, feat: Any) -> Tensor:
        # Feat is tuple (feat_low, feat_high)
        feat_low, feat_high = feat
        logits_low = self.low_expert.forward_head(feat_low)
        logits_high = self.high_expert.forward_head(feat_high)
        # Default simple mean if only the generic feature/head API is used.
        return 0.5 * (logits_low + logits_high)

    def forward_from_features(
        self,
        waveform: Tensor | None,
        spectrogram: Tensor,
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Run both experts once and route from acoustic plus block-9 features."""
        if waveform is None:
            # Spectrogram-only compatibility path; at deployment, pass waveform
            # so the router also receives the six physical acoustic features.
            spectral_feat = spectrogram.new_zeros((spectrogram.shape[0], 6))
        else:
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            spectral_feat = self.acoustic_extractor(waveform)

        logits_low, context_low = self.low_expert.forward_with_context(spectrogram, context_block=9)
        logits_high, context_high = self.high_expert.forward_with_context(spectrogram, context_block=9)
        router_token = 0.5 * (context_low + context_high)
        alpha, gates = self.router(spectral_feat, router_token)
        logits = (1.0 - alpha) * logits_low + alpha * logits_high
        if return_details:
            return {
                "logits": logits,
                "logits_low": logits_low,
                "logits_high": logits_high,
                "alpha": alpha,
                "gates": gates,
                "router_token": router_token,
            }
        return logits

    def forward(
        self,
        x: Tensor,
        snr_db: Tensor | None = None,
        mixture: Tensor | None = None,
        return_details: bool = False,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """
        Dual-Expert Forward Pass with MoE-v2 self-contained routing:
          x: Waveform [B, 64000] or Spectrogram [B, 1, 64, 398]
          snr_db and mixture: accepted for API compatibility but never used by the router.
        """
        if x.dim() in (1, 2):
            raw_wave = x.unsqueeze(0) if x.dim() == 1 else x
            spec = self.extract_features_from_wave(raw_wave)
        else:
            raw_wave = None
            spec = x
        return self.forward_from_features(raw_wave, spec, return_details=return_details)
