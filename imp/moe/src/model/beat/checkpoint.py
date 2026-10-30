# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .beat import BEATClassifier

FOLD_PREFIX = "encoder.model."


def load_fold_teacher(
    path: str | Path, config_path: str | Path, device: torch.device
) -> tuple[BEATClassifier, dict[str, Any]]:
    """
    Nạp mô hình BEATs Teacher (12 layers, 90.34M tham số) từ file checkpoint.
    
    Hỗ trợ 2 định dạng checkpoint:
        1. Standalone EMA checkpoint (ví dụ: beats_teacher_ema_standalone.pt):
           Chứa state_dict dạng "backbone.*" đã được trích xuất sẵn.
        2. Fold checkpoint (ví dụ: beats_teacher_fold0_weights.pt):
           Chứa tiền tố FOLD_PREFIX "encoder.model.*" kèm trọng số "head.weight", "head.bias".
           Hàm sẽ tự động bóc tách tiền tố, ánh xạ lại các layer FFN/PatchEmbed và gán vào head.
           
    Sau khi nạp, mô hình Teacher được chuyển sang device đích và đóng băng tham số (freeze)
    để phục vụ quá trình chưng cất tri thức hoặc suy luận MoE.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    teacher = BEATClassifier.from_yaml(config_path)

    # Trường hợp 1: Checkpoint dạng standalone EMA
    if isinstance(checkpoint, dict) and "model" in checkpoint and any(k.startswith("backbone.") for k in checkpoint["model"]):
        sd = checkpoint["model"]
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in sd.items()}
        result = teacher.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"{path}: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
        metadata = {"epoch": checkpoint.get("best_epoch", "ema"), "format": "standalone_ema"}
        return teacher.to(device).freeze(), metadata

    # Trường hợp 2: Checkpoint dạng Huấn luyện Fold nguyên bản (kèm prefix encoder.model.)
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
