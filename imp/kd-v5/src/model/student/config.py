from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig


@dataclass(frozen=True)
class StudentConfig(BaseConfig):
    section: ClassVar[str] = "student"
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

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.freq_bins % 16 or self.frames < 16:
            raise ValueError("freq_bins must be divisible by 16 and frames must be at least 16")
        if self.init not in {"ced-tiny", "imagenet", "none"}:
            raise ValueError(f"unknown init {self.init!r}")
        if self.bias_init not in {"zero"}:
            raise ValueError(f"unknown bias_init {self.bias_init!r}")
        for name in ("dropout", "attn_dropout", "drop_path"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
