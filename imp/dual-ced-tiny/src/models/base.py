"""
Base Classifier Module for Dual-CED-Tiny:
Provides abstract classifier interface handling both raw waveforms and spectrograms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.frontend import LogMelFrontend


class BaseAudioClassifier(nn.Module, ABC):
    """
    Abstract base class for all audio classification models in Dual-CED-Tiny.
    Accepts both:
      1. Raw waveform: [B, 64000] (or [64000]) -> automatically processed via LogMelFrontend.
      2. Precomputed spectrogram: [B, 1, F, T].
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 64,
        num_frames: int = 398,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.num_frames = num_frames

        self.frontend = LogMelFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )

    def extract_features_from_wave(self, wave: Tensor) -> Tensor:
        """Transforms raw waveform [B, T_samples] to spectrogram [B, 1, n_mels, num_frames]."""
        return self.frontend(wave)

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        """Extracts latent representation from spectrogram [B, 1, F, T]."""
        raise NotImplementedError

    @abstractmethod
    def forward_head(self, feat: Tensor) -> Tensor:
        """Projects feature representation to logits [B, num_classes]."""
        raise NotImplementedError

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """
        Universal forward pass:
          If 1D/2D waveform -> converts to spectrogram first.
          If 4D spectrogram -> passes directly to forward_features.
        """
        if x.dim() in (1, 2):
            x = self.extract_features_from_wave(x)
        feat = self.forward_features(x)
        return self.forward_head(feat)

    def count_parameters(self) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
