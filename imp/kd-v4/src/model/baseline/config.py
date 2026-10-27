from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig


@dataclass(frozen=True)
class BaselineConfig(BaseConfig):
    section: ClassVar[str] = "model"

    arch: str = "resnet18"
    label: str = ""
    family: str = ""
    num_classes: int = 40
    freq_bins: int = 128
    num_frames: int = 960
    pretrained: bool = False
    width_mult: float = 1.0
    channels: tuple[int, ...] = (64, 128, 256, 512)
    hidden: int = 448
    layers: int = 3
    heads: int = 4
    ffn_ratio: int = 4
    kernel_size: int = 31
    time_pool: int = 4
    dropout: float = 0.1
    window_size: tuple[int, int] = (4, 6)
    specaug_freq_masks: int = 2
    specaug_freq_width: int = 16
    specaug_time_masks: int = 2
    specaug_time_width: int = 24
