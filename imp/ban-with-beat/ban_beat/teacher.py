from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model.beat import BEATClassifier
from src.model.frcrn import FRCRNSeparator

from .frontend import apply_front_end
from .runtime import autocast_context

FOLD_INPUT_RMS = 0.07925
DEFAULT_INPUT_RMS = 0.05


@dataclass(frozen=True)
class TeacherMetadata:
    path: Path
    format: str
    input_rms: float | None
    uses_frcrn: bool
    epoch: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "input_rms": self.input_rms,
            "uses_frcrn": self.uses_frcrn,
            "epoch": self.epoch,
        }


class BANTeacher(nn.Module):
    def __init__(
        self,
        classifier: BEATClassifier,
        separator: FRCRNSeparator | None = None,
        input_rms: float | None = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.separator = separator
        self.input_rms = input_rms
        self.config = classifier.config

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        return self.classifier(apply_front_end(waveform, self.separator, self.input_rms)).logits

    @torch.no_grad()
    def forward_with_hidden(self, waveform: Tensor, layers: Sequence[int]) -> tuple[Tensor, tuple[Tensor, ...]]:
        output = self.classifier(apply_front_end(waveform, self.separator, self.input_rms), layers=layers)
        return output.logits, output.hidden

    def freeze(self) -> BANTeacher:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def _load_beats_fairseq(classifier: BEATClassifier, state: Mapping[str, Tensor], prefix: str) -> None:
    backbone = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not backbone:
        raise ValueError(f"checkpoint has no BEATs weights with prefix {prefix!r}")
    converted = classifier.convert_checkpoint({"model": backbone})
    for key, value in state.items():
        if key.startswith("head."):
            converted[f"head.fc.{key[len('head.') :]}"] = value
    report = classifier.load_state_dict(converted, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise ValueError(
            f"BEATs load mismatch: missing {list(report.missing_keys[:5])}, unexpected {list(report.unexpected_keys[:5])}"
        )


def _load_frcrn(checkpoint: Mapping[str, Any], frcrn_config: str | Path) -> FRCRNSeparator:
    state = checkpoint["state_dict"]
    base = FRCRNSeparator.from_yaml(frcrn_config)
    separator = FRCRNSeparator(replace(base.config, **checkpoint.get("frcrn_config", {})))
    frcrn_state = {key[len("frcrn.") :]: value for key, value in state.items() if key.startswith("frcrn.")}
    report = separator.load_state_dict(frcrn_state, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise ValueError(
            f"FRCRN load mismatch: missing {list(report.missing_keys[:5])}, unexpected {list(report.unexpected_keys[:5])}"
        )
    return separator


def load_teacher(
    checkpoint_path: str | Path,
    model_config: str | Path,
    frcrn_config: str | Path,
    device: torch.device,
    input_rms: float | None = None,
) -> tuple[BANTeacher, TeacherMetadata]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    classifier = BEATClassifier.from_yaml(model_config)
    separator: FRCRNSeparator | None = None
    resolved_rms = input_rms
    uses_frcrn = False

    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        if any(key.startswith("frcrn.") for key in state):
            separator = _load_frcrn(checkpoint, frcrn_config)
            _load_beats_fairseq(classifier, state, "encoder.")
            uses_frcrn = True
            resolved_rms = (
                resolved_rms if resolved_rms is not None else float(checkpoint.get("input_rms", DEFAULT_INPUT_RMS))
            )
            checkpoint_format = "fairseq-frcrn"
        elif any(key.startswith("encoder.model.") for key in state):
            _load_beats_fairseq(classifier, state, "encoder.model.")
            resolved_rms = resolved_rms if resolved_rms is not None else FOLD_INPUT_RMS
            checkpoint_format = "fairseq-fold"
        elif any(key.startswith("encoder.") for key in state):
            _load_beats_fairseq(classifier, state, "encoder.")
            resolved_rms = (
                resolved_rms if resolved_rms is not None else float(checkpoint.get("input_rms", DEFAULT_INPUT_RMS))
            )
            checkpoint_format = "fairseq"
        else:
            raise ValueError(f"unsupported teacher checkpoint keys in {path}: {list(state)[:5]}")
        epoch = checkpoint.get("epoch", checkpoint.get("best_epoch"))
    elif "model" in checkpoint:
        classifier.load_state_dict(checkpoint["model"])
        checkpoint_format = "ban"
        epoch = checkpoint.get("epoch")
    else:
        classifier.load_state_dict(checkpoint)
        checkpoint_format = "state-dict"
        epoch = None

    teacher = BANTeacher(classifier, separator, resolved_rms).to(device).freeze()
    metadata = TeacherMetadata(
        path=path,
        format=checkpoint_format,
        input_rms=resolved_rms,
        uses_frcrn=uses_frcrn,
        epoch=int(epoch) if epoch is not None else None,
    )
    return teacher, metadata


def teacher_fingerprint(checkpoint_path: str | Path, extra_paths: Sequence[str | Path] = ()) -> str:
    parts = []
    for value in (checkpoint_path, *extra_paths):
        path = Path(value)
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
        else:
            parts.append(f"{path}|missing")
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]


def save_teacher_cache(
    path: str | Path,
    sample_ids: Sequence[str],
    logits: np.ndarray,
    fingerprint: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(list(sample_ids)),
        logits=np.asarray(logits, dtype=np.float32),
        fingerprint=np.asarray([fingerprint]),
    )
    return output


def load_teacher_cache(
    path: str | Path,
    fingerprint: str,
    expected_size: int | None = None,
) -> dict[str, Tensor] | None:
    source = Path(path)
    if not source.is_file():
        return None
    with np.load(source) as data:
        if str(data["fingerprint"].item()) != fingerprint:
            return None
        sample_ids = [str(item) for item in data["sample_ids"].tolist()]
        logits = np.asarray(data["logits"], dtype=np.float32)
    if expected_size is not None and len(sample_ids) != expected_size:
        return None
    return {
        sample_id: torch.from_numpy(np.asarray(row, dtype=np.float32))
        for sample_id, row in zip(sample_ids, logits, strict=True)
    }


@torch.no_grad()
def cache_teacher_logits(
    teacher: BANTeacher,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    desc: str = "teacher cache",
) -> tuple[list[str], np.ndarray]:
    teacher.eval()
    sample_ids: list[str] = []
    chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        waveform = batch["waveform"].to(device)
        with autocast_context(device, amp):
            logits = teacher(waveform)
        sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
        chunks.append(logits.float().cpu().numpy())
    if not chunks:
        raise ValueError("loader produced no batches")
    return sample_ids, np.concatenate(chunks, axis=0)


def ensure_teacher_cache(
    teacher: BANTeacher,
    loaders: Mapping[str, DataLoader],
    datasets: Mapping[str, Any],
    cache_dir: str | Path,
    fingerprint: str,
    device: torch.device,
    amp: str,
    persist: bool,
) -> dict[str, dict[str, Tensor]]:
    directory = Path(cache_dir)
    if persist:
        directory.mkdir(parents=True, exist_ok=True)
    caches: dict[str, dict[str, Tensor]] = {}
    for split, loader in loaders.items():
        expected = len(datasets[split])
        path = directory / f"teacher_{split}.npz"
        loaded = load_teacher_cache(path, fingerprint, expected) if persist else None
        if loaded is not None:
            caches[split] = loaded
            continue
        sample_ids, logits = cache_teacher_logits(teacher, loader, device, amp, desc=f"teacher cache {split}")
        if len(sample_ids) != expected:
            raise ValueError(f"{split}: cached {len(sample_ids)} samples, expected {expected}")
        if persist:
            save_teacher_cache(path, sample_ids, logits, fingerprint)
        caches[split] = {
            sample_id: torch.from_numpy(np.asarray(row, dtype=np.float32))
            for sample_id, row in zip(sample_ids, logits, strict=True)
        }
    return caches
