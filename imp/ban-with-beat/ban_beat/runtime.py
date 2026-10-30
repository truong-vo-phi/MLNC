from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor, nn

RMS_EPS = 1e-8


def resolve_device(name: str | None = None) -> torch.device:
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@contextmanager
def autocast_context(device: torch.device, amp: str) -> Iterator[None]:
    enabled = amp == "bf16" and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        yield


def parameters_m(model: nn.Module) -> float:
    return round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 3)


def rms_normalize(waveform: Tensor, target_rms: float) -> Tensor:
    scale = target_rms / (waveform.square().mean(dim=-1, keepdim=True).sqrt() + RMS_EPS)
    return waveform * scale


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
