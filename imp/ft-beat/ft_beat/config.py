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
    "macro_f1",
    "top1_accuracy",
    "balanced_accuracy",
    "mAP",
)
AMP_MODES = ("bf16", "fp16", "none")


@dataclass(frozen=True)
class FTBeatFRCRNConfig(BaseConfig):
    section: ClassVar[str] = "frcrn"

    checkpoint: str = "pretrained/FRCRN/best_e037_20261020T203958Z.pt"
    fallback_checkpoint: str = "pretrained/frcrn_best.pt"
    target_rms: float = 0.07925
    freeze: bool = True

    def __post_init__(self) -> None:
        if self.target_rms <= 0:
            raise ValueError(f"target_rms must be positive, got {self.target_rms}")


@dataclass(frozen=True)
class FTBeatBEATsConfig(BaseConfig):
    section: ClassVar[str] = "beats"

    checkpoint: str = "pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    fallback_checkpoint: str = "pretrained/BEATs_iter3_plus_AS2M.pt"


@dataclass(frozen=True)
class FTBeatDataConfig(BaseConfig):
    section: ClassVar[str] = "data"

    dataset_key: str = "mixture-36l"
    duration: float = 4.0
    snr_levels: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    num_workers: int = 4
    balanced_sampling: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    nhat_cache_dir: str | None = None
    audioset_eval_segments: str | None = None
    audioset_train_segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if not self.snr_levels:
            raise ValueError("snr_levels must not be empty")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.nhat_cache_dir is not None and not self.nhat_cache_dir.strip():
            raise ValueError("nhat_cache_dir must not be empty when set")
        for name in ("max_train_samples", "max_eval_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")


@dataclass(frozen=True)
class FTBeatTrainConfig(BEATTrainConfig):
    section: ClassVar[str] = "train"

    epochs: int = 10
    patience: int = 4
    batch_size: int = 16
    accumulation_steps: int = 2
    eval_batch_size: int = 64
    amp: str = "bf16"
    seed: int = 2026
    resume_every: int = 1
    selection_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps must be positive, got {self.accumulation_steps}")
        if self.amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {self.amp!r}")
        if self.resume_every <= 0:
            raise ValueError(f"resume_every must be positive, got {self.resume_every}")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}, got {self.selection_metric!r}")


@dataclass(frozen=True)
class FTBeatConfig:
    model_config: Path
    frcrn_config: Path
    frcrn: FTBeatFRCRNConfig
    beats: FTBeatBEATsConfig
    data: FTBeatDataConfig
    train: FTBeatTrainConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": str(self.model_config),
            "frcrn_config": str(self.frcrn_config),
            "frcrn": self.frcrn.to_dict(),
            "beats": self.beats.to_dict(),
            "data": self.data.to_dict(),
            "train": self.train.to_dict(),
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


def load_config(path: str | Path) -> FTBeatConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    model_config = raw.get("model_config")
    frcrn_config = raw.get("frcrn_config")
    if not model_config:
        raise ValueError(f"model_config is missing in {config_path}")
    if not frcrn_config:
        raise ValueError(f"frcrn_config is missing in {config_path}")
    return FTBeatConfig(
        model_config=resolve_path(model_config),
        frcrn_config=resolve_path(frcrn_config),
        frcrn=FTBeatFRCRNConfig.from_dict(raw.get("frcrn", {})),
        beats=FTBeatBEATsConfig.from_dict(raw.get("beats", {})),
        data=FTBeatDataConfig.from_dict(raw.get("data", {})),
        train=FTBeatTrainConfig.from_dict(raw.get("train", {})),
    )


def smoke_config(config: FTBeatConfig) -> FTBeatConfig:
    return replace(
        config,
        data=replace(config.data, num_workers=0, max_train_samples=32, max_eval_samples=32),
        train=replace(
            config.train,
            epochs=1,
            patience=1,
            batch_size=4,
            accumulation_steps=1,
            eval_batch_size=4,
            warmup_ratio=0.0,
        ),
    )
