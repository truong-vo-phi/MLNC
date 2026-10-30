from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

from src.core import BaseConfig

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def resolve_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("path cannot be None")
    p = Path(path)
    if p.is_absolute():
        return p
    for root in (PACKAGE_ROOT, REPO_ROOT):
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    return (PACKAGE_ROOT / p).resolve()


@dataclass(frozen=True)
class CEDFRCRNConfig(BaseConfig):
    section: ClassVar[str] = "frcrn"
    checkpoint: str = "pretrained/FRCRN/best_e037_20261020T203958Z.pt"
    fallback_checkpoint: str = "pretrained/frcrn_best.pt"
    target_rms: float = 0.07925
    freeze: bool = True

    def __post_init__(self) -> None:
        if self.target_rms <= 0:
            raise ValueError("target_rms must be positive")


@dataclass(frozen=True)
class CEDTinyConfig(BaseConfig):
    section: ClassVar[str] = "ced_tiny"
    arch: str = "deit_tiny_patch16_224"
    num_classes: int = 36
    freq_bins: int = 64
    frames: int = 398
    init: str = "ced-tiny"
    ced_repo: str = "mispeech/ced-tiny"
    ced_path: str | None = None
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.0
    post_ln: bool = False
    pre_ln: bool = True
    bias_init: str = "zero"
    use_bias: bool = True
    time_major: bool = False
    hybrid_pool: bool = False

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.freq_bins % 16 != 0 or self.frames < 16:
            raise ValueError("freq_bins must be divisible by 16 and frames must be at least 16")
        if self.init not in {"ced-tiny", "imagenet", "none"}:
            raise ValueError(f"unknown init {self.init!r}")
        if self.bias_init not in {"normal", "zero", "random"}:
            raise ValueError(f"unknown bias_init {self.bias_init!r}")
        for name in ("dropout", "attn_dropout", "drop_path"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class FrontendConfig(BaseConfig):
    section: ClassVar[str] = "frontend"
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float = 8000.0
    num_frames: int = 398
    log_eps: float = 1.0e-6
    norm_eps: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.n_mels <= 0 or self.num_frames <= 0:
            raise ValueError("sample_rate, n_mels and num_frames must be positive")


@dataclass(frozen=True)
class SpecAugmentConfig(BaseConfig):
    section: ClassVar[str] = "specaugment"
    freq_masks: int = 2
    freq_width: int = 8
    time_masks: int = 2
    time_width: int = 24

    def __post_init__(self) -> None:
        if self.freq_masks < 0 or self.time_masks < 0 or self.freq_width < 0 or self.time_width < 0:
            raise ValueError("mask counts and widths must be non-negative")


@dataclass(frozen=True)
class CEDDataConfig(BaseConfig):
    section: ClassVar[str] = "data"
    dataset_key: str = "mixture-36l"
    duration: float = 4.0
    snr_levels: list[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    num_workers: int = 4
    balanced_sampling: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    nhat_cache_dir: str | None = None
    audioset_eval_segments: str | None = None
    audioset_train_segments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class CEDTrainConfig(BaseConfig):
    section: ClassVar[str] = "train"
    optimizer: str = "adamw"
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    warmup_steps: int = 500
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    label_smoothing: float = 0.05
    mixup: float = 0.5
    mixup_alpha: float = 0.4
    grad_clip: float = 1.0
    epochs: int = 15
    patience: int = 7
    batch_size: int = 32
    accumulation_steps: int = 1
    eval_batch_size: int = 64
    amp: str = "bf16"
    seed: int = 2026
    resume_every: int = 1
    selection_metric: str = "macro_acc"

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.mixup < 0.0:
            raise ValueError("mixup must be non-negative")


@dataclass(frozen=True)
class CEDTinyPipelineConfig(BaseConfig):
    frcrn_config: str = "config/frcrn.yaml"
    frcrn: CEDFRCRNConfig = field(default_factory=CEDFRCRNConfig)
    ced_tiny: CEDTinyConfig = field(default_factory=CEDTinyConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    data: CEDDataConfig = field(default_factory=CEDDataConfig)
    train: CEDTrainConfig = field(default_factory=CEDTrainConfig)


def load_config(path: str | Path) -> CEDTinyPipelineConfig:
    p = resolve_path(path)
    with open(p, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    frcrn_cfg = CEDFRCRNConfig.from_dict(raw.get(CEDFRCRNConfig.section, {}))
    ced_cfg = CEDTinyConfig.from_dict(raw.get(CEDTinyConfig.section, {}))
    frontend_cfg = FrontendConfig.from_dict(raw.get(FrontendConfig.section, {}))
    spec_cfg = SpecAugmentConfig.from_dict(raw.get(SpecAugmentConfig.section, {}))
    data_cfg = CEDDataConfig.from_dict(raw.get(CEDDataConfig.section, {}))
    train_cfg = CEDTrainConfig.from_dict(raw.get(CEDTrainConfig.section, {}))

    return CEDTinyPipelineConfig(
        frcrn_config=raw.get("frcrn_config", "config/frcrn.yaml"),
        frcrn=frcrn_cfg,
        ced_tiny=ced_cfg,
        frontend=frontend_cfg,
        specaugment=spec_cfg,
        data=data_cfg,
        train=train_cfg,
    )


def smoke_config(config: CEDTinyPipelineConfig) -> CEDTinyPipelineConfig:
    data_dict = config.data.to_dict()
    data_dict.update(
        {
            "num_workers": 0,
            "max_train_samples": 32,
            "max_eval_samples": 16,
        }
    )
    train_dict = config.train.to_dict()
    train_dict.update(
        {
            "epochs": 1,
            "batch_size": 4,
            "accumulation_steps": 1,
            "eval_batch_size": 4,
            "warmup_steps": 2,
        }
    )
    ced_dict = config.ced_tiny.to_dict()
    ced_dict.update(
        {
            "init": "none",
        }
    )
    return CEDTinyPipelineConfig(
        frcrn_config=config.frcrn_config,
        frcrn=copy.deepcopy(config.frcrn),
        ced_tiny=CEDTinyConfig.from_dict(ced_dict),
        frontend=copy.deepcopy(config.frontend),
        specaugment=copy.deepcopy(config.specaugment),
        data=CEDDataConfig.from_dict(data_dict),
        train=CEDTrainConfig.from_dict(train_dict),
    )
