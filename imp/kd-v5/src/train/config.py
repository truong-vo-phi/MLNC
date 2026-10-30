from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig


@dataclass(frozen=True)
class DatasetConfig(BaseConfig):
    section: ClassVar[str] = "dataset"
    name: str = "mixture-36l"
    duration: float = 4.0
    noise_cache: str | None = None
    train_sample_ids: str | None = None
    validation_sample_ids: str | None = None
    test_sample_ids: str | None = None
    snr_db: tuple[float, ...] | None = None


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    section: ClassVar[str] = "train"
    epochs: int = 30
    patience: int = 7
    lr: float = 1.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 500
    lr_patience: int = 1
    lr_decay: float = 0.5
    min_lr: float = 1.0e-6
    grad_clip: float = 1.0
    batch_size: int = 32
    eval_batch_size: int = 64
    mixup: float = 0.8
    mixup_alpha: float = 0.4
    gap_penalty: float = 0.2
    train_eval_clips: int = 2000
    num_workers: int = 4
    seed: int = 42
    selection_metric: str = "score"
    lr_schedule: str = "cosine"
    amp: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if not 0.0 <= self.mixup <= 1.0:
            raise ValueError("mixup must be in [0, 1]")
        if self.mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be positive")
        if self.lr_patience < 1:
            raise ValueError("lr_patience must be at least 1")


@dataclass(frozen=True)
class KDTrainConfig(TrainConfig):
    pass


@dataclass(frozen=True)
class KDLossConfig(BaseConfig):
    section: ClassVar[str] = "loss"
    temperature: float = 4.0
    weight: float = 1.0
    label_smoothing: float = 0.1

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
