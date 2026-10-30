from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Literal

import torch

AMPMode = Literal["bf16", "fp16", "none"]


def resolve_device(device_str: str | None = None) -> torch.device:
    """Xác định device chạy huấn luyện (CUDA nếu có, ngược lại CPU)."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def autocast_context(device: torch.device, amp: str = "bf16") -> Iterator[None]:
    """Ngữ cảnh autocast an toàn cho CUDA (bf16/fp16) và CPU."""
    device_type = device.type
    if device_type == "cuda" and amp in ("bf16", "fp16"):
        dtype = torch.bfloat16 if amp == "bf16" else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            yield
    elif device_type == "cpu" and amp == "bf16":
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            yield
    else:
        yield
