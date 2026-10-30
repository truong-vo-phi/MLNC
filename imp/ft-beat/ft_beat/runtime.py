from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def resolve_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def autocast_context(device: torch.device, amp_mode: str) -> Iterator[None]:
    if amp_mode == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    elif amp_mode in ("bf16", "fp16") and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    elif amp_mode == "bf16" and device.type == "cpu":
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            yield
    else:
        yield
