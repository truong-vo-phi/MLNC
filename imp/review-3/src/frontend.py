"""
Audio Frontend: Log-Mel Filterbank and SpecAugment for Review-3.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torchvision

# Scoped compatibility patch for torchaudio on Python 3.13 / PyTorch 2.11
if hasattr(torch, "_ops") and hasattr(torch._ops, "_OpNamespace"):
    _orig_op_getattr = torch._ops._OpNamespace.__getattr__
    def _safe_op_getattr(self, op_name):
        try:
            return _orig_op_getattr(self, op_name)
        except AttributeError:
            if getattr(self, "name", None) == "torchaudio":
                return lambda *args, **kwargs: None
            raise
    torch._ops._OpNamespace.__getattr__ = _safe_op_getattr

import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch import Tensor


class FrontendOutput(NamedTuple):
    features: Tensor
    lengths: Tensor | None = None


class LogMel(nn.Module):
    """
    Log-Mel Spectrogram Frontend.
    Converts raw 16kHz audio waveforms (B, num_samples) to Log-Mel Spectrogram (B, 1, n_mels, num_frames).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        num_frames: int = 398,
        log_eps: float = 1.0e-6,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_frames = num_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps

        self.mel_scale = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
        )

    def forward(self, waveform: Tensor) -> FrontendOutput:
        """
        Args:
            waveform: (B, num_samples) or (B, 1, num_samples)
        Returns:
            FrontendOutput with features shaped (B, 1, n_mels, num_frames)
        """
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)

        mel = self.mel_scale(waveform)  # (B, n_mels, T_raw)
        log_mel = torch.log(mel + self.log_eps)

        # Instance normalization over time & freq per sample
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True)
        norm_mel = (log_mel - mean) / (std + self.norm_eps)

        # Ensure consistent time frame length
        cur_frames = norm_mel.shape[-1]
        if cur_frames < self.num_frames:
            norm_mel = F.pad(norm_mel, (0, self.num_frames - cur_frames))
        elif cur_frames > self.num_frames:
            norm_mel = norm_mel[..., : self.num_frames]

        features = norm_mel.unsqueeze(1)  # (B, 1, n_mels, num_frames)
        return FrontendOutput(features=features)


class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation: Frequency masking and Time masking.
    """

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 16,
        time_masks: int = 2,
        time_width: int = 24,
    ) -> None:
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, 1, F, T)
        """
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x

        b, c, f, t = x.shape
        out = x.clone()

        # Frequency masking
        for _ in range(self.freq_masks):
            w = torch.randint(0, min(self.freq_width, f), (b,), device=x.device)
            f0 = torch.randint(0, f, (b,), device=x.device)
            for i in range(b):
                if w[i] > 0:
                    start = min(f0[i].item(), f - w[i].item())
                    out[i, :, start : start + w[i].item(), :] = 0.0

        # Time masking
        for _ in range(self.time_masks):
            w = torch.randint(0, min(self.time_width, t), (b,), device=x.device)
            t0 = torch.randint(0, t, (b,), device=x.device)
            for i in range(b):
                if w[i] > 0:
                    start = min(t0[i].item(), t - w[i].item())
                    out[i, :, :, start : start + w[i].item()] = 0.0

        return out
