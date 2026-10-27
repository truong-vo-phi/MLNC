from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig

RESIZE_MODES = ("nearest", "area")


@dataclass(frozen=True)
class ReviewKDConfig(BaseConfig):
    section: ClassVar[str] = "review"

    teacher_layers: tuple[int, ...] = (3, 6, 9, 12)
    teacher_dim: int = 768
    mid_channels: int = 512
    resize: str = "area"
    target_rms: float = 0.0
    noise_from_residual: bool | None = None

    def __post_init__(self) -> None:
        if self.resize not in RESIZE_MODES:
            raise ValueError(f"resize must be one of {RESIZE_MODES}, got {self.resize!r}")
        if list(self.teacher_layers) != sorted(set(self.teacher_layers)):
            raise ValueError(f"teacher_layers must be strictly increasing, got {self.teacher_layers}")
        if self.target_rms < 0:
            raise ValueError(f"target_rms must be >= 0, got {self.target_rms}")
