from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

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


FRCRN_SELECTION_METRICS = ("noise_si_snr", "noise_si_snr_hi", "speech_si_snr", "speech_si_snr_hi")


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
    batch_candidates: tuple[int, ...] = ()
    vram_fraction: float = 0.9
    lr_decay: float = 1.0
    selection_metric: str = "noise_si_snr"
    high_snr_levels: tuple[float, ...] = (15.0, 20.0)

    def __post_init__(self) -> None:
        if self.selection_metric not in FRCRN_SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {FRCRN_SELECTION_METRICS}, got {self.selection_metric!r}"
            )
        if not 0.0 < self.lr_decay <= 1.0:
            raise ValueError(f"lr_decay must be in (0, 1], got {self.lr_decay}")


@dataclass(frozen=True)
class BEATTrainConfig(TrainConfig):
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    label_smoothing: float = 0.0
    plateau_factor: float = 0.0
    plateau_patience: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.plateau_factor < 1.0:
            raise ValueError(f"plateau_factor must be in [0, 1), got {self.plateau_factor}")
        if self.plateau_patience <= 0:
            raise ValueError(f"plateau_patience must be positive, got {self.plateau_patience}")


@dataclass(frozen=True)
class KDMoETrainConfig(TrainConfig):
    lr: float = 5e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 120
    patience: int = 15
    stage1_epochs: int = 8
    ema_decay: float = 0.999
    amp: str = "bf16"
    seed: int = 42
    resume_every: int = 1


@dataclass(frozen=True)
class ReviewKDTrainConfig(BEATTrainConfig):
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 100
    patience: int = 15
    review_weight: float = 1.0
    review_warmup_epochs: int = 5
    hcl_levels: tuple[int, ...] = (4, 2, 1)
    logit_weight: float = 0.0  # KD trên logit teacher (softmax, nhiệt độ); 0 = tắt
    logit_temperature: float = 4.0

    def __post_init__(self) -> None:
        if self.review_weight < 0:
            raise ValueError(f"review_weight must be >= 0, got {self.review_weight}")
        if self.review_warmup_epochs < 0:
            raise ValueError(f"review_warmup_epochs must be >= 0, got {self.review_warmup_epochs}")
        if self.logit_weight < 0:
            raise ValueError(f"logit_weight must be >= 0, got {self.logit_weight}")
        if self.logit_temperature <= 0:
            raise ValueError(f"logit_temperature must be > 0, got {self.logit_temperature}")
