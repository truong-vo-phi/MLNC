from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.beat import BEATClassifier
from src.model.frcrn import FRCRNSeparator

from .config import ReviewKDConfig

RMS_EPS = 1e-8


def rms_normalize(waveform: Tensor, target_rms: float) -> Tensor:
    rms = waveform.square().mean(dim=-1).sqrt()
    return waveform * (target_rms / (rms + RMS_EPS))[:, None]


def tokens_to_grid(tokens: Tensor, freq_bins: int) -> Tensor:
    batch, count, dim = tokens.shape
    if count % freq_bins:
        raise ValueError(f"{count} tokens do not divide into {freq_bins} frequency patches")
    return tokens.reshape(batch, count // freq_bins, freq_bins, dim).permute(0, 3, 2, 1).contiguous()


class ReviewTeacher(nn.Module):
    def __init__(
        self, separator: FRCRNSeparator, beats: BEATClassifier, config: ReviewKDConfig, target_rms: float
    ) -> None:
        super().__init__()
        if target_rms <= 0:
            raise ValueError(f"target_rms must be positive, got {target_rms}")
        if config.noise_from_residual is None:
            raise ValueError("noise_from_residual must be resolved to True or False before building the teacher")
        self.separator = separator.freeze()
        self.beats = beats.freeze()
        self.config = config
        self.target_rms = target_rms
        self.freq_bins = beats.frontend.num_mel_bins // beats.config.input_patch_size

    def train(self, mode: bool = True) -> ReviewTeacher:
        return super().train(False)

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        estimate = self.separator(waveform)
        if self.config.noise_from_residual:
            estimate = waveform - estimate
        output = self.beats(rms_normalize(estimate, self.target_rms), layers=self.config.teacher_layers)
        return torch.stack([tokens_to_grid(hidden, self.freq_bins) for hidden in output.hidden])
