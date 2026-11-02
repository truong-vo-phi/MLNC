"""
Dataset Loading and Memory-Mapped Cache Access for Mixture-36.

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
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
    t0_logits: Tensor
    tema_logits: Tensor


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
        if labels_path.is_file():
            with open(labels_path, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = [f"class_{i}" for i in range(36)]

        manifest_dir = "eval" if split in ("eval", "validation") else split
        manifest_path = self.root / manifest_dir / "manifest.csv"
        if not manifest_path.is_file():
            manifest_path = self.root / split / "manifest.csv"

        if manifest_path.is_file():
            self.rows = read_csv(manifest_path)
        else:
            self.rows = []

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
    Memory-mapped dataset connecting the FRCRN-separated noise cache (.npy)
    with the manifest metadata.
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

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> AudioItem:
        item = self.base[index]
        waveform = torch.from_numpy(self._array[index].copy()).float()
        item["waveform"] = waveform
        return item


class KDNoiseCacheDataset(Dataset):
    """
    Cache dataset with precomputed teacher logits for high-speed offline KD distillation.
    """
    def __init__(
        self,
        base: Mixture36Dataset,
        npy_path: str | Path,
        t0_logits_path: str | Path | None = None,
        tema_logits_path: str | Path | None = None,
    ) -> None:
        self.base = base
        self.npy_path = Path(npy_path)
        self._array = np.load(self.npy_path, mmap_mode="r")

        self.t0_logits = np.load(t0_logits_path, mmap_mode="r") if t0_logits_path else None
        self.tema_logits = np.load(tema_logits_path, mmap_mode="r") if tema_logits_path else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> AudioItem:
        item = self.base[index]
        item["waveform"] = torch.from_numpy(self._array[index].copy()).float()
        if self.t0_logits is not None:
            item["t0_logits"] = torch.from_numpy(self.t0_logits[index].copy()).float()
        if self.tema_logits is not None:
            item["tema_logits"] = torch.from_numpy(self.tema_logits[index].copy()).float()
        return item


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
