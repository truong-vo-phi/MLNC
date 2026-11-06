"""
BEATs-12L teacher utilities for logit-only Knowledge Distillation.

Teachers are frozen and run inference only: their logits are computed once per dataset and cached on disk,
so the training graph only contains the two CED-Tiny experts.

  * Low-SNR expert  <- BEATs fold0 teacher   (beats_teacher_fold0_weights.pt)
  * High-SNR expert <- BEATs EMA  teacher    (beats_teacher_ema_standalone.pt)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import src.beats as _beats_pkg
from src.beats.beat import BEATClassifier
from src.dataset import rms_normalize

logger = logging.getLogger(__name__)

FOLD_PREFIX = "encoder.model."


def _register_legacy_aliases() -> None:
    """Teacher checkpoints were pickled with the original `src.model.beat.*` module paths."""
    for name in ("src.model", "src.model.beat", "src.model.beat.beat"):
        sys.modules.setdefault(name, _beats_pkg)


def resolve_teacher_path(
    local_path: str | Path | None,
    hf_file: str | None,
    hf_repo: str | None,
    root: Path,
) -> Path:
    """Use the local checkpoint when present, otherwise download `hf_file` from the HF dataset repo."""
    if local_path:
        p = Path(local_path)
        p = p if p.is_absolute() else (root / p).resolve()
        if p.is_file():
            return p
        logger.warning("Teacher checkpoint not found locally at %s; falling back to Hugging Face.", p)
    if not (hf_file and hf_repo):
        raise FileNotFoundError(f"Teacher checkpoint unavailable: local={local_path}, hf={hf_repo}/{hf_file}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(hf_repo, hf_file, repo_type="dataset"))


def load_beats_teacher(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
) -> tuple[BEATClassifier, dict[str, Any]]:
    """Loads a frozen BEATs teacher from a fold checkpoint (`encoder.model.*`) or a standalone EMA checkpoint."""
    _register_legacy_aliases()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = BEATClassifier.from_yaml(config_path)

    if isinstance(ckpt, dict) and "model" in ckpt and any(k.startswith("backbone.") for k in ckpt["model"]):
        fmt = "standalone_ema"
        state = {k: v.float() if torch.is_tensor(v) else v for k, v in ckpt["model"].items()}
        result = teacher.load_state_dict(state, strict=False)
    else:
        fmt = "fold"
        sd = ckpt.get("state_dict")
        if sd is None or not any(k.startswith(FOLD_PREFIX) for k in sd):
            raise ValueError(f"{checkpoint_path}: expected '{FOLD_PREFIX}*' keys (BEATs fold checkpoint)")
        backbone = {k[len(FOLD_PREFIX):]: v for k, v in sd.items() if k.startswith(FOLD_PREFIX)}
        converted = teacher.convert_checkpoint({"model": backbone})
        converted["head.fc.weight"] = sd["head.weight"]
        converted["head.fc.bias"] = sd["head.bias"]
        result = teacher.load_state_dict(converted, strict=False)

    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{checkpoint_path}: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")

    for p in teacher.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in teacher.parameters())
    return teacher.to(device).eval(), {"format": fmt, "params": n_params}


@torch.no_grad()
def compute_teacher_logits(
    teacher: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    cache_path: str | Path | None = None,
    input_rms: float = 0.07925,
    amp: bool = True,
) -> np.ndarray:
    """Teacher logits [N, C] over `dataset` (row-aligned), cached to `cache_path` (.npy) when given.

    The waveform is RMS-normalized to `input_rms` before the BEATs frontend, matching the
    reference pipeline (`imp/moe`). The FRCRN cache itself is not normalized (mean RMS ~0.04),
    so skipping this step feeds the teachers a distribution they were not trained on.
    """
    if cache_path is not None and Path(cache_path).is_file():
        arr = np.load(cache_path)
        if len(arr) == len(dataset):
            logger.info("Loaded cached teacher logits %s %s", Path(cache_path).name, arr.shape)
            return arr

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    use_amp = amp and device.type == "cuda"
    for batch in loader:
        wave = rms_normalize(batch["waveform"].to(device), target_rms=input_rms)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = teacher(wave).logits
        out.append(logits.float().cpu().numpy())
    arr = np.concatenate(out, axis=0)
    if not np.isfinite(arr).all():
        raise FloatingPointError("Non-finite teacher logits")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr
