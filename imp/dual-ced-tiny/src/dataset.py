"""
Dataset Loader for Mixture Audio across Multiple SNR levels in Dual-CED-Tiny.
Supports real disk manifests (train/manifest.csv, eval/manifest.csv) and synthetic fallback.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence, TypedDict

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
TARGET_RMS = 0.07925
RMS_EPS = 1.0e-8


def rms_normalize(waveform: Tensor, target_rms: float = TARGET_RMS, eps: float = RMS_EPS) -> Tensor:
    """
    Standardize the energy level of waveform to target_rms (default 0.07925).
    Formula: x_norm = x * (target_rms / (sqrt(mean(x^2)) + eps))
    """
    if waveform.dim() == 1:
        rms = torch.sqrt(torch.mean(waveform ** 2) + eps)
        return waveform * (target_rms / rms)
    rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True) + eps)
    return waveform * (target_rms / rms)


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    target_idx: Tensor
    snr_db: Tensor
    sample_id: str


def read_csv_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class MixtureDataset(Dataset):
    """
    Dataset loader for mixture audio on clean mix-dataset-v2.
    Loads real WAV files referenced by train/manifest.csv and eval/manifest.csv.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_classes: int = 36,
        snr_levels: Sequence[float] = SNR_LEVELS,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = "eval" if split in ("eval", "val", "validation") else split
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.num_classes = num_classes
        self.snr_levels = tuple(snr_levels)

        manifest_path = self.root / self.split / "manifest.csv"
        if not manifest_path.is_file():
            # Alternative layout: check root / manifest.csv
            manifest_path = self.root / f"{self.split}_manifest.csv"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        rows = read_csv_manifest(manifest_path)
        if self.snr_levels:
            allowed = set(self.snr_levels)
            rows = [
                r for r in rows
                if float(r.get("target_snr_db", r.get("achieved_snr_db", 0.0))) in allowed
            ]

        if max_samples and max_samples < len(rows):
            rows = rows[:max_samples]

        self.rows = rows

        # Load class names if present
        labels_path = self.root / "labels.txt"
        if labels_path.is_file():
            with open(labels_path, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> AudioItem:
        row = self.rows[idx]
        sample_id = row.get("sample_id", f"sample_{idx}")

        # Locate audio file
        audio_file = row.get("mixture_path") or row.get("file_path") or row.get("path")
        if not audio_file:
            audio_file = self.root / self.split / "mixture" / f"{sample_id}.wav"
        else:
            audio_file = Path(audio_file)
            if not audio_file.is_absolute():
                audio_file = self.root / audio_file

        if audio_file.is_file():
            wave, sr = sf.read(str(audio_file), dtype="float32")
            if sr != self.sample_rate:
                # Basic resample fallback if required
                wave = np.interp(
                    np.linspace(0, len(wave), int(len(wave) * self.sample_rate / sr)),
                    np.arange(len(wave)),
                    wave,
                )
        else:
            # Fallback zero waveform if file not found
            wave = np.zeros(self.num_samples, dtype=np.float32)

        # Pad or crop to exact num_samples (64000)
        if len(wave) < self.num_samples:
            pad = np.zeros(self.num_samples - len(wave), dtype=np.float32)
            wave = np.concatenate([wave, pad])
        elif len(wave) > self.num_samples:
            wave = wave[: self.num_samples]

        wave_tensor = torch.from_numpy(wave).float()

        # Target label
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if "multi_hot_36" in row:
            raw_target = json.loads(row["multi_hot_36"])
            target = torch.tensor(raw_target, dtype=torch.float32)
            target_idx = torch.argmax(target)
        elif "target_idx" in row:
            idx_val = int(row["target_idx"])
            target[idx_val] = 1.0
            target_idx = torch.tensor(idx_val, dtype=torch.long)
        else:
            target_idx = torch.tensor(0, dtype=torch.long)
            target[0] = 1.0

        snr_val = float(row.get("target_snr_db", row.get("achieved_snr_db", 0.0)))
        snr_tensor = torch.tensor(snr_val, dtype=torch.float32)

        return {
            "waveform": wave_tensor,
            "target": target,
            "target_idx": target_idx,
            "snr_db": snr_tensor,
            "sample_id": sample_id,
            "index": idx,
        }


class FrcrnCacheDataset(Dataset):
    """
    FRCRN-denoised residual waveforms (memory-mapped cache .npy, row-aligned with the manifest).
    This is the classifier input used by the mix_dataset_v2 pipeline and by the BEATs teachers.
    """

    def __init__(
        self,
        root: str | Path,
        cache_path: str | Path,
        split: str = "train",
        num_classes: int = 36,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = "eval" if split in ("eval", "val", "validation") else split
        self.num_classes = num_classes

        manifest_path = self.root / self.split / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        if not Path(cache_path).is_file():
            raise FileNotFoundError(f"FRCRN cache not found at {cache_path}")

        rows = read_csv_manifest(manifest_path)
        self.array = np.load(cache_path, mmap_mode="r")
        if len(rows) != len(self.array):
            raise ValueError(f"Manifest/cache size mismatch: {len(rows)} rows vs {len(self.array)} cached waveforms")

        targets = np.array([int(np.argmax(json.loads(r["multi_hot_36"]))) for r in rows], dtype=np.int64)
        snrs = np.array([float(r.get("target_snr_db", r.get("achieved_snr_db", 0.0))) for r in rows], dtype=np.float32)
        ids = [r.get("sample_id", f"sample_{i}") for i, r in enumerate(rows)]

        if max_samples and max_samples < len(rows):  # strided subset keeps every SNR level represented
            self.indices = np.linspace(0, len(rows) - 1, max_samples).astype(np.int64)
        else:
            self.indices = np.arange(len(rows), dtype=np.int64)
        self.targets = targets[self.indices]
        self.snrs = snrs[self.indices]
        self.ids = [ids[i] for i in self.indices]

        labels_path = self.root / "labels.txt"
        if labels_path.is_file():
            with open(labels_path, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> AudioItem:
        j = int(self.indices[idx])
        target_idx = int(self.targets[idx])
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[target_idx] = 1.0
        return {
            "waveform": torch.from_numpy(np.array(self.array[j], dtype=np.float32)),
            "target": target,
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "snr_db": torch.tensor(float(self.snrs[idx]), dtype=torch.float32),
            "sample_id": self.ids[idx],
            "index": idx,
        }


class SyntheticMixtureDataset(Dataset):
    """
    Synthetic dataset generating controlled dummy audio tensors for testing and smoke verification.
    """

    def __init__(
        self,
        num_samples_total: int = 64,
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_classes: int = 36,
        snr_levels: Sequence[float] = SNR_LEVELS,
    ) -> None:
        self.num_samples_total = num_samples_total
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(duration * sample_rate)
        self.num_classes = num_classes
        self.snr_levels = tuple(snr_levels)
        self.class_names = [f"synthetic_class_{i}" for i in range(num_classes)]

        rng = np.random.default_rng(2026)
        self.data_snrs = [float(rng.choice(self.snr_levels)) for _ in range(num_samples_total)]
        self.data_labels = [int(rng.integers(0, num_classes)) for _ in range(num_samples_total)]

    def __len__(self) -> int:
        return self.num_samples_total

    def __getitem__(self, idx: int) -> AudioItem:
        rng = np.random.default_rng(2026 + idx)
        wave = rng.standard_normal(self.num_samples).astype(np.float32) * 0.1
        wave_tensor = torch.from_numpy(wave)

        label_idx = self.data_labels[idx]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[label_idx] = 1.0

        return {
            "waveform": wave_tensor,
            "target": target,
            "target_idx": torch.tensor(label_idx, dtype=torch.long),
            "snr_db": torch.tensor(self.data_snrs[idx], dtype=torch.float32),
            "sample_id": f"synth_{idx:05d}",
            "index": idx,
        }


def get_dataset(
    config: dict[str, Any],
    split: str = "train",
    use_synthetic: bool = False,
    max_samples: int | None = None,
) -> Dataset:
    dataset_cfg = config.get("dataset", {})
    if use_synthetic:
        return SyntheticMixtureDataset(
            num_samples_total=max_samples or (64 if split == "train" else 32),
            sample_rate=dataset_cfg.get("sample_rate", 16000),
            duration=dataset_cfg.get("duration", 4.0),
            num_classes=dataset_cfg.get("num_classes", 36),
            snr_levels=dataset_cfg.get("snr_levels", SNR_LEVELS),
        )

    data_root = dataset_cfg.get("data_root", "datasets/mix-dataset-v2")

    frcrn_root = dataset_cfg.get("frcrn_root")
    if frcrn_root:
        cache_by_split = {
            "train": "train.npy",
            "eval": "validation.npy",
            "val": "validation.npy",
            "validation": "validation.npy",
            "test": "test.npy",
        }
        cache_name = cache_by_split.get(split)
        cache_path = Path(frcrn_root) / cache_name if cache_name else None
        if cache_path is not None and cache_path.is_file():
            return FrcrnCacheDataset(
                root=data_root,
                cache_path=cache_path,
                split=split,
                num_classes=dataset_cfg.get("num_classes", 36),
                max_samples=max_samples,
            )
        if cache_path is not None:
            logger.warning("FRCRN cache %s not found; falling back to raw mixture WAVs.", cache_path)

    try:
        return MixtureDataset(
            root=data_root,
            split=split,
            sample_rate=dataset_cfg.get("sample_rate", 16000),
            duration=dataset_cfg.get("duration", 4.0),
            num_classes=dataset_cfg.get("num_classes", 36),
            snr_levels=dataset_cfg.get("snr_levels", SNR_LEVELS),
            max_samples=max_samples,
        )
    except FileNotFoundError:
        logger.warning(
            "Dataset manifest not found at %s. Falling back to SyntheticMixtureDataset.",
            data_root,
        )
        return SyntheticMixtureDataset(
            num_samples_total=max_samples or (64 if split == "train" else 32),
            sample_rate=dataset_cfg.get("sample_rate", 16000),
            duration=dataset_cfg.get("duration", 4.0),
            num_classes=dataset_cfg.get("num_classes", 36),
            snr_levels=dataset_cfg.get("snr_levels", SNR_LEVELS),
        )


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
