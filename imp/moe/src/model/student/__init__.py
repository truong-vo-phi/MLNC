from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import StudentConfig
from .frontend import LogMel, LogMelConfig, SpecAugment, SpecAugmentConfig
from .model import CEDStudent, StudentOutput
from .vit_bias import ViTSBias, load_ced_tiny


def build_student(
    raw: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> CEDStudent:
    values = {**raw["student"], **(overrides or {})}
    ced_path = values.get("ced_path")
    if ced_path and not Path(ced_path).is_absolute() and root is not None:
        values["ced_path"] = str(root / ced_path)
    student = StudentConfig.from_dict(values)
    frontend = LogMelConfig.from_dict(raw["frontend"])
    spec_augment = SpecAugmentConfig.from_dict(raw["specaugment"])
    return CEDStudent(student, LogMel(**frontend.to_dict()), SpecAugment(**spec_augment.to_dict()))


__all__ = [
    "CEDStudent",
    "LogMel",
    "LogMelConfig",
    "SpecAugment",
    "SpecAugmentConfig",
    "StudentConfig",
    "StudentOutput",
    "ViTSBias",
    "build_student",
    "load_ced_tiny",
]
