from __future__ import annotations

from torch import Tensor, nn

from .runtime import rms_normalize


def apply_front_end(waveform: Tensor, separator: nn.Module | None, input_rms: float | None) -> Tensor:
    features = separator(waveform) if separator is not None else waveform
    if input_rms is not None:
        features = rms_normalize(features, input_rms)
    return features
