from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import yaml

from src.core import BaseConfig
from src.train import BEATTrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SELECTION_METRICS = (
    "macro_f1_snr",
    "top1_accuracy_snr",
    "macro_f1",
    "top1_accuracy",
    "macro_f1_hi",
    "top1_accuracy_hi",
)
AMP_MODES = ("bf16", "none")


@dataclass(frozen=True)
class BANDataConfig(BaseConfig):
    section: ClassVar[str] = "data"

    dataset_key: str = "mixture-36l"
    duration: float = 4.0
    snr_levels: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    num_workers: int = 4
    balanced_sampling: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    nhat_cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.nhat_cache_dir is not None and not self.nhat_cache_dir.strip():
            raise ValueError("nhat_cache_dir must not be empty when set")
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
class BANLossConfig(BaseConfig):
    section: ClassVar[str] = "loss"

    kd_weight: float = 0.8
    temperature: float = 1.0
    confidence_weighting: bool = False
    confidence_power: float = 1.0
    attention_weight: float = 0.0
    attention_layers: tuple[int, ...] = (4, 8, 12)

    def __post_init__(self) -> None:
        if not 0.0 <= self.kd_weight <= 1.0:
            raise ValueError(f"kd_weight must be in [0, 1], got {self.kd_weight}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.confidence_power < 0:
            raise ValueError(f"confidence_power must be >= 0, got {self.confidence_power}")
        if self.attention_weight < 0:
            raise ValueError(f"attention_weight must be >= 0, got {self.attention_weight}")
        if self.attention_weight > 0 and not self.attention_layers:
            raise ValueError("attention_layers must not be empty when attention_weight > 0")
        if any(not 1 <= layer <= 12 for layer in self.attention_layers):
            raise ValueError(f"attention_layers must be in 1..12, got {self.attention_layers}")


@dataclass(frozen=True)
class BANTeacherConfig(BaseConfig):
    section: ClassVar[str] = "teacher"

    checkpoint: str | None = None
    input_rms: float | None = None
    cache_logits: bool = True
    cache_dir: str = "cache/ban-beat/teacher"
    frcrn_config: str = "config/frcrn.yaml"
    finetune: bool = False
    frcrn_checkpoint: str | None = None
    reuse: bool = False

    def __post_init__(self) -> None:
        if self.finetune and not self.frcrn_checkpoint:
            raise ValueError("teacher.frcrn_checkpoint is required when teacher.finetune is true")
        if self.input_rms is not None and self.input_rms <= 0:
            raise ValueError(f"input_rms must be positive when set, got {self.input_rms}")
        if not self.cache_dir.strip():
            raise ValueError("cache_dir must not be empty")
        if not self.frcrn_config.strip():
            raise ValueError("frcrn_config must not be empty")


@dataclass(frozen=True)
class BANTrainConfig(BEATTrainConfig):
    section: ClassVar[str] = "train"

    epochs: int = 100
    patience: int = 15
    batch_size: int = 32
    eval_batch_size: int = 64
    amp: str = "bf16"
    seed: int = 42
    resume_every: int = 1
    selection_metric: str = "macro_f1_snr"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {self.amp!r}")
        if self.resume_every <= 0:
            raise ValueError(f"resume_every must be positive, got {self.resume_every}")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}, got {self.selection_metric!r}")


@dataclass(frozen=True)
class BANGenerationConfig(BaseConfig):
    section: ClassVar[str] = "generations"

    count: int = 2
    student_init: str = "pretrained"
    ensemble: bool = True
    seeds: tuple[int, ...] = (42, 43, 44)

    def __post_init__(self) -> None:
        if self.count < 1 or self.count > 5:
            raise ValueError(f"count must be in [1, 5], got {self.count}")
        if self.student_init not in ("pretrained", "scratch"):
            raise ValueError(f"student_init must be 'pretrained' or 'scratch', got {self.student_init!r}")
        if len(self.seeds) < self.count:
            raise ValueError(f"need at least {self.count} seeds, got {len(self.seeds)}")


@dataclass(frozen=True)
class BANBaselineConfig(BaseConfig):
    section: ClassVar[str] = "baseline"

    enabled: bool = False
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class BANConfig:
    model_config: Path
    data: BANDataConfig
    teacher: BANTeacherConfig
    loss: BANLossConfig
    train: BANTrainConfig
    generations: BANGenerationConfig
    baseline: BANBaselineConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": str(self.model_config),
            "data": self.data.to_dict(),
            "teacher": self.teacher.to_dict(),
            "loss": self.loss.to_dict(),
            "train": self.train.to_dict(),
            "generations": self.generations.to_dict(),
            "baseline": self.baseline.to_dict(),
        }


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    package_path = PACKAGE_ROOT / path
    return package_path if package_path.exists() else repo_path


def load_config(path: str | Path) -> BANConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    model_config = raw.get("model_config")
    if not model_config:
        raise ValueError(f"model_config is missing in {config_path}")
    return BANConfig(
        model_config=resolve_path(model_config),
        data=BANDataConfig.from_dict(raw.get("data", {})),
        teacher=BANTeacherConfig.from_dict(raw.get("teacher", {})),
        loss=BANLossConfig.from_dict(raw.get("loss", {})),
        train=BANTrainConfig.from_dict(raw.get("train", {})),
        generations=BANGenerationConfig.from_dict(raw.get("generations", {})),
        baseline=BANBaselineConfig.from_dict(raw.get("baseline", {})),
    )


def smoke_config(config: BANConfig) -> BANConfig:
    return replace(
        config,
        data=replace(config.data, num_workers=0, max_train_samples=64, max_eval_samples=64),
        teacher=replace(config.teacher, cache_logits=False),
        train=replace(
            config.train,
            epochs=1,
            patience=1,
            batch_size=8,
            eval_batch_size=8,
            warmup_ratio=0.0,
        ),
        generations=replace(config.generations, count=1, ensemble=False),
        baseline=replace(config.baseline, enabled=False),
    )
