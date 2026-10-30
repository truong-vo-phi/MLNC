from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .beat import BEATClassifier

FOLD_PREFIX = "encoder.model."


def load_fold_teacher(
    path: str | Path, config_path: str | Path, device: torch.device
) -> tuple[BEATClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    teacher = BEATClassifier.from_yaml(config_path)

    if isinstance(checkpoint, dict) and "model" in checkpoint and any(k.startswith("backbone.") for k in checkpoint["model"]):
        sd = checkpoint["model"]
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in sd.items()}
        result = teacher.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"{path}: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
        metadata = {"epoch": checkpoint.get("best_epoch", "ema"), "format": "standalone_ema"}
        return teacher.to(device).freeze(), metadata

    state = checkpoint.get("state_dict")
    if state is None or not any(key.startswith(FOLD_PREFIX) for key in state):
        raise ValueError(f"{path}: expected a BEATs fold checkpoint with '{FOLD_PREFIX}*' keys")
    backbone = {key[len(FOLD_PREFIX) :]: value for key, value in state.items() if key.startswith(FOLD_PREFIX)}
    converted = teacher.convert_checkpoint({"model": backbone})
    converted["head.fc.weight"] = state["head.weight"]
    converted["head.fc.bias"] = state["head.bias"]
    result = teacher.load_state_dict(converted, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{path}: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
    metadata = {"epoch": checkpoint.get("best_epoch"), "format": "fold"}
    return teacher.to(device).freeze(), metadata
