from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import yaml

from src.core import BaseConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

SELECTION_METRICS = (
    "macro_f1",
    "top1_accuracy",
    "macro_f1_snr",
    "top1_accuracy_snr",
    "mAP",
)
SELECTION_TARGETS = ("ema", "model", "both")
AMP_MODES = ("bf16", "none")


@dataclass(frozen=True)
class EMAConfig(BaseConfig):
    """Cấu hình cho Exponential Moving Average (EMA) kết hợp CRD."""

    section: ClassVar[str] = "ema"

    decay: float = 0.9995
    warmup_steps: int = 1000
    warmup_decay_start: float = 0.90
    update_interval: int = 1
    cpu_offload: bool = False
    evaluate_ema: bool = True
    selection_target: str = "ema"  # "ema", "model", hoặc "both"
    kd_weight: float = 1.0
    temperature: float = 4.0
    crd_weight: float = 0.8
    crd_dim: int = 128
    crd_negatives: int = 4096
    crd_temperature: float = 0.07
    crd_momentum: float = 0.5
    crd_num_samples: int = 30240

    def __post_init__(self) -> None:
        if not (0.0 < self.decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        if not (0.0 <= self.warmup_decay_start <= self.decay):
            raise ValueError(
                f"warmup_decay_start ({self.warmup_decay_start}) must be in [0, decay={self.decay}]"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {self.update_interval}")
        if self.kd_weight < 0:
            raise ValueError(f"kd_weight must be >= 0, got {self.kd_weight}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.selection_target not in SELECTION_TARGETS:
            raise ValueError(
                f"selection_target must be one of {SELECTION_TARGETS}, got {self.selection_target!r}"
            )
        if self.crd_weight < 0:
            raise ValueError(f"crd_weight must be >= 0, got {self.crd_weight}")
        if self.crd_dim <= 0:
            raise ValueError(f"crd_dim must be positive, got {self.crd_dim}")
        if self.crd_negatives <= 0:
            raise ValueError(f"crd_negatives must be positive, got {self.crd_negatives}")
        if self.crd_temperature <= 0:
            raise ValueError(f"crd_temperature must be positive, got {self.crd_temperature}")
        if not (0.0 <= self.crd_momentum < 1.0):
            raise ValueError(f"crd_momentum must be in [0, 1), got {self.crd_momentum}")
        if self.crd_num_samples <= 0:
            raise ValueError(f"crd_num_samples must be positive, got {self.crd_num_samples}")


@dataclass(frozen=True)
class EMABeatDataConfig(BaseConfig):
    """Cấu hình dữ liệu cho huấn luyện EMA-BEATs."""

    section: ClassVar[str] = "data"

    dataset_key: str = "mixture-36l"
    cache_nhat_dir: str = "dataset/cache"
    audioset_dir: str = "/marimo/datasets/audioset/AudioSet"
    duration: float = 4.0
    snr_levels: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    num_workers: int = 4
    balanced_sampling: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not self.snr_levels:
            raise ValueError("snr_levels must not be empty")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        for name in ("max_train_samples", "max_eval_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")


@dataclass(frozen=True)
class EMABeatTrainConfig(BaseConfig):
    """Cấu hình tối ưu hóa & vòng lặp huấn luyện."""

    section: ClassVar[str] = "train"

    optimizer: str = "adamw"
    lr: float = 1.0e-4
    layer_decay: float = 0.3
    min_lr: float = 1.0e-6
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    scheduler: str = "plateau"
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    label_smoothing: float = 0.0
    grad_clip: float = 5.0
    epochs: int = 25
    patience: int = 5
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    batch_size: int = 32
    eval_batch_size: int = 64
    amp: str = "bf16"
    seed: int = 42
    resume_every: int = 1
    selection_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not (0.0 < self.layer_decay <= 1.0):
            raise ValueError(f"layer_decay must be in (0, 1], got {self.layer_decay}")
        if self.min_lr <= 0:
            raise ValueError(f"min_lr must be positive, got {self.min_lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if len(self.betas) != 2 or not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError(f"betas must be a pair of floats in [0, 1), got {self.betas}")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if not (0.0 <= self.plateau_factor < 1.0):
            raise ValueError(f"plateau_factor must be in [0, 1), got {self.plateau_factor}")
        if self.plateau_patience <= 0:
            raise ValueError(f"plateau_patience must be positive, got {self.plateau_patience}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be positive, got {self.eval_batch_size}")
        if self.amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {self.amp!r}")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {SELECTION_METRICS}, got {self.selection_metric!r}"
            )
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be positive, got {self.eval_batch_size}")
        if self.amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {self.amp!r}")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {SELECTION_METRICS}, got {self.selection_metric!r}"
            )


@dataclass(frozen=True)
class EMABeatConfig(BaseConfig):
    """Cấu hình hoàn chỉnh cho thí nghiệm EMA-BEATs."""

    model_config: str = "config/beat.yaml"
    data: EMABeatDataConfig = EMABeatDataConfig()
    ema: EMAConfig = EMAConfig()
    train: EMABeatTrainConfig = EMABeatTrainConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "data": self.data.to_dict(),
            "ema": self.ema.to_dict(),
            "train": self.train.to_dict(),
        }


def _resolve_config_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        for base in (Path.cwd(), PACKAGE_ROOT, REPO_ROOT):
            candidate = (base / target).resolve()
            if candidate.is_file():
                return candidate
    return target


def load_config(path: str | Path) -> EMABeatConfig:
    """Nạp file cấu hình yaml và khởi tạo EMABeatConfig."""
    resolved = _resolve_config_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {path} (resolved to {resolved})")

    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    model_config = data.get("model_config", "config/beat.yaml")

    data_section = data.get("data", {})
    if "snr_levels" in data_section and isinstance(data_section["snr_levels"], list):
        data_section["snr_levels"] = tuple(float(x) for x in data_section["snr_levels"])

    train_section = data.get("train", {})
    if "betas" in train_section and isinstance(train_section["betas"], list):
        train_section["betas"] = tuple(float(x) for x in train_section["betas"])

    return EMABeatConfig(
        model_config=model_config,
        data=EMABeatDataConfig(**data_section),
        ema=EMAConfig(**data.get("ema", {})),
        train=EMABeatTrainConfig(**train_section),
    )


def smoke_config(config: EMABeatConfig) -> EMABeatConfig:
    """Tạo phiên bản cấu hình smoke test thu nhỏ (1 epoch, batch nhỏ, ít mẫu) để kiểm tra pipeline."""
    return replace(
        config,
        data=replace(
            config.data,
            num_workers=0,
            max_train_samples=64,
            max_eval_samples=32,
        ),
        ema=replace(
            config.ema,
            warmup_steps=4,
            update_interval=1,
            crd_num_samples=64,
            crd_negatives=16,
        ),
        train=replace(
            config.train,
            epochs=1,
            batch_size=8,
            eval_batch_size=8,
            patience=1,
            warmup_ratio=0.1,
        ),
    )
