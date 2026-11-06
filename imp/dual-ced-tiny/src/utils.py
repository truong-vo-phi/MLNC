"""
Utility Functions for Dual-CED-Tiny:
Logging, Random Seeding, Checkpointing, and Parameter Counting.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 2026) -> None:
    """Sets deterministic seed across Python, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "dual_ced_tiny",
    log_dir: str | Path = "logs",
) -> logging.Logger:
    """Configures formatted dual-output logger (.log file + stdout console)."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info("Initialized logger: outputting to %s", log_file)
    return logger


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns total parameter count and trainable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> Path:
    """Atomically saves checkpoint dictionary to disk with best model mirroring."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / filename
    temp_target = target.with_suffix(".tmp.pt")

    torch.save(state, temp_target)
    temp_target.replace(target)

    if is_best:
        best_target = ckpt_dir / "best_model.pt"
        torch.save(state, best_target)

    return target


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Loads weights and state from checkpoint path."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    elif "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_state_dict(payload)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload
