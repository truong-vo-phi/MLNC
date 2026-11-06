"""
Multi-crop overlap inference for Dual-CED-Tiny.

Replaces the older 3-way temporal TTA (±50 ms) with several overlapping windows inside each clip.
For a 4.0 s mixture and the default settings, the window starts are 0.0 / 0.4 / 0.8 s for 3.2 s crops,
so the whole clip is covered without padding outside its boundaries. The per-window logits are averaged.

The waveform passed to :func:`overlap_logits` must already be preprocessed exactly like the training input
(RMS-normalized to ``kd.input_rms``); this module only slices and aggregates.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

DEFAULT_NUM_WINDOWS = 3
DEFAULT_CROP_SECONDS = 3.2


def overlap_crop_starts(
    num_samples: int,
    num_windows: int = DEFAULT_NUM_WINDOWS,
    crop_samples: int | None = None,
) -> list[int]:
    """
    Returns evenly spaced crop start offsets (in samples), including both endpoints when possible.

    Args:
        num_samples: Total number of samples in the clip.
        num_windows: Requested number of crops (>= 1). Duplicate starts are collapsed.
        crop_samples: Crop length in samples. Defaults to the full clip (single window).

    Examples:
        >>> overlap_crop_starts(64000, num_windows=3, crop_samples=51200)
        [0, 6400, 12800]
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")

    if crop_samples is None:
        crop_samples = num_samples
    crop_samples = int(crop_samples)
    if crop_samples <= 0:
        raise ValueError(f"crop_samples must be positive, got {crop_samples}")
    crop_samples = min(crop_samples, num_samples)

    max_start = num_samples - crop_samples
    if num_windows == 1 or max_start == 0:
        return [0]
    if num_windows == 2:
        return [0, max_start]

    starts: list[int] = []
    for i in range(num_windows):
        start = int(round(i * max_start / (num_windows - 1)))
        if not starts or start != starts[-1]:
            starts.append(start)
    return starts


@torch.no_grad()
def overlap_logits(
    model: Any,
    wave: Tensor,
    snr_db: Tensor | None = None,
    num_windows: int = DEFAULT_NUM_WINDOWS,
    crop_seconds: float = DEFAULT_CROP_SECONDS,
    sample_rate: int = 16000,
    aggregate: str = "logits",
) -> Tensor:
    """
    Runs the model on overlapping crops and aggregates the predictions.

Args:
        model: Callable accepting ``waveform[B, T]`` and returning logits ``[B, C]``.
        wave: RMS-normalized waveform ``[B, T]`` or ``[T]``.
        snr_db: Legacy compatibility argument; the learned router does not consume it.
        num_windows: Number of overlapping crops.
        crop_seconds: Crop duration in seconds.
        sample_rate: Sampling rate used to convert ``crop_seconds`` to samples.
        aggregate: ``"logits"`` averages raw logits; ``"probs"`` averages softmax probabilities.

    Returns:
        Aggregated predictions ``[B, C]`` in float32.
    """
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    if wave.dim() != 2:
        raise ValueError(f"wave must be [B, T] or [T], got shape {tuple(wave.shape)}")

    aggregate = aggregate.lower()
    if aggregate not in ("logits", "probs"):
        raise ValueError(f"aggregate must be 'logits' or 'probs', got {aggregate!r}")

    crop_seconds = float(crop_seconds)
    if crop_seconds <= 0:
        raise ValueError(f"crop_seconds must be positive, got {crop_seconds}")

    total_samples = int(wave.shape[-1])
    crop_samples = min(max(1, int(round(crop_seconds * sample_rate))), total_samples)
    starts = overlap_crop_starts(total_samples, num_windows=num_windows, crop_samples=crop_samples)

    per_window: list[Tensor] = []
    for start in starts:
        chunk = wave[..., start : start + crop_samples]
        if chunk.shape[-1] < crop_samples:
            chunk = torch.nn.functional.pad(chunk, (0, crop_samples - chunk.shape[-1]))
        logits = model(chunk, snr_db=snr_db)
        per_window.append(logits.float())

    stacked = torch.stack(per_window, dim=0)  # [W, B, C]
    if aggregate == "probs":
        return torch.softmax(stacked, dim=-1).mean(dim=0)
    return stacked.mean(dim=0)
