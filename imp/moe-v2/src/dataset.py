"""
Dataset Loading and Memory-Mapped Cache Access for Mixture-36.

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16000


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    target_idx: Tensor
    snr_db: Tensor
    sample_id: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class Mixture36Dataset(Dataset):
    """
    Dataset representing the 36-class acoustic environment mixture corpus.
    Manifest contains: sample_id, split, target_snr_db, multi_hot_36, etc.
    """
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float = 4.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)

        labels_path = self.root / "labels.txt"
        with open(labels_path, encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        manifest_dir = "eval" if split in ("eval", "validation") else split
        manifest_path = self.root / manifest_dir / "manifest.csv"
        if not manifest_path.is_file():
            manifest_path = self.root / split / "manifest.csv"

        self.rows = read_csv(manifest_path)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        target_vec = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
        target_idx = target_vec.argmax(dim=-1)
        snr_val = float(row.get("target_snr_db", row.get("achieved_snr_db", 0.0)))
        return {
            "target": target_vec,
            "target_idx": target_idx,
            "snr_db": torch.tensor(snr_val, dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


class NoiseCacheDataset(Dataset):
    """
    High-performance zero-copy memory-mapped dataset connecting the FRCRN-separated
    noise cache (.npy) with the manifest metadata.
    """
    def __init__(self, base: Mixture36Dataset, npy_path: str | Path) -> None:
        self.base = base
        self.npy_path = Path(npy_path)
        if not self.npy_path.is_file():
            raise FileNotFoundError(f"Cache file not found: {self.npy_path}")

        self._array = np.load(self.npy_path, mmap_mode="r")
        if len(self._array) != len(base):
            raise ValueError(f"Cache size mismatch: {len(self._array)} vs {len(base)}")

    @property
    def class_names(self) -> list[str]:
        return self.base.class_names

    @property
    def num_classes(self) -> int:
        return self.base.num_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> AudioItem:
        base_item = self.base[index]
        # Copy from mmap slice to ensure thread-safety and continuous memory layout
        wave_np = np.array(self._array[index], dtype=np.float32)
        waveform = torch.from_numpy(wave_np)
        return {
            "waveform": waveform,
            "target": base_item["target"],
            "target_idx": base_item["target_idx"],
            "snr_db": base_item["snr_db"],
            "sample_id": base_item["sample_id"],
        }


class KDNoiseCacheDataset(NoiseCacheDataset):
    """
    Dataset extending NoiseCacheDataset with Dual-Teacher precomputed logits
    for high-speed Knowledge Distillation training.
    """
    def __init__(
        self,
        base: Mixture36Dataset,
        npy_path: str | Path,
        t0_logits_path: str | Path | None = None,
        tema_logits_path: str | Path | None = None,
    ) -> None:
        super().__init__(base, npy_path)
        self.t0_logits = None
        self.tema_logits = None

        if t0_logits_path is not None and Path(t0_logits_path).is_file():
            self.t0_logits = np.load(t0_logits_path, mmap_mode="r")
            if len(self.t0_logits) != len(base):
                raise ValueError(f"Teacher 0 logits length mismatch: {len(self.t0_logits)} vs {len(base)}")

        if tema_logits_path is not None and Path(tema_logits_path).is_file():
            self.tema_logits = np.load(tema_logits_path, mmap_mode="r")
            if len(self.tema_logits) != len(base):
                raise ValueError(f"Teacher EMA logits length mismatch: {len(self.tema_logits)} vs {len(base)}")

    def __getitem__(self, index: int) -> AudioItem:
        item = super().__getitem__(index)
        if self.t0_logits is not None:
            item["t0_logits"] = torch.from_numpy(np.array(self.t0_logits[index], dtype=np.float32))
        if self.tema_logits is not None:
            item["tema_logits"] = torch.from_numpy(np.array(self.tema_logits[index], dtype=np.float32))
        return item


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
