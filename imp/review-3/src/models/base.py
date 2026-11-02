"""
Base Audio Classifier Wrapper for Review-3 Models.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.frontend import LogMel, SpecAugment


class BaseAudioClassifier(nn.Module, abc.ABC):
    """
    Unified Base Audio Classifier.
    Supports receiving both raw 1D audio waveforms (B, num_samples) and 4D mel-spectrograms (B, 1, F, T).
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        specaug_freq_masks: int = 2,
        specaug_freq_width: int = 16,
        specaug_time_masks: int = 2,
        specaug_time_width: int = 24,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.num_frames = num_frames

        self.frontend = LogMel(
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.spec_augment = SpecAugment(
            freq_masks=specaug_freq_masks,
            freq_width=specaug_freq_width,
            time_masks=specaug_time_masks,
            time_width=specaug_time_width,
        )

    @abc.abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        """Forward through backbone on spectrogram (B, 1, F, T)."""
        pass

    @abc.abstractmethod
    def forward_head(self, feat: Tensor) -> Tensor:
        """Forward through classifier head to produce logits (B, num_classes)."""
        pass

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x: Raw waveform (B, T) or Spectrogram (B, 1, F, T)
            return_features: If True, returns (logits, feat)
        Returns:
            logits: (B, num_classes) or (logits, feat)
        """
        if x.dim() <= 2:
            spec = self.frontend(x).features
        elif x.dim() == 3:
            spec = x.unsqueeze(1)
        else:
            spec = x

        if self.training:
            spec = self.spec_augment(spec)

        feat = self.forward_features(spec)
        logits = self.forward_head(feat)
        if return_features:
            return logits, feat
        return logits

    def count_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

