from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from src.core import BaseConfig


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    section: ClassVar[str] = "train"
    epochs: int = 30
    patience: int = 5
    lr: float = 5e-5
    grad_clip: float = 5.0
    batch_size: int = 32
    eval_batch_size: int = 64

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        normalized = dict(values)
        for name in ("betas", "snr_levels"):
            if name in normalized:
                normalized[name] = tuple(normalized[name])
        return super().from_dict(normalized)


@dataclass(frozen=True)
class FRCRNTrainConfig(TrainConfig):
    epochs: int = 20
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 12
    eval_batch_size: int = 32
    halve_lr_after: int = 3
    segment_seconds: float = 2.0
    eval_segment_seconds: float = 4.0
    snr_levels: tuple[float, ...] = (-5, 0, 5, 10, 15, 20)


@dataclass(frozen=True)
class BEATTrainConfig(TrainConfig):
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.0
