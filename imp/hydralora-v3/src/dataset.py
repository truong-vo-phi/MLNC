"""
Nạp dữ liệu và Truy xuất Cache ánh xạ bộ nhớ (Memory-Mapped Cache) cho Tập dữ liệu Mixture-36.

Tài liệu tham khảo:
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
    t0_logits: Tensor
    tema_logits: Tensor


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class Mixture36Dataset(Dataset):
    """
    Tập dữ liệu đại diện cho kho ngữ liệu âm thanh 36 lớp (Mixture-36).
    Manifest chứa: sample_id, split, target_snr_db, multi_hot_36, v.v.
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        sample_id = row.get("sample_id", f"sample_{idx}")
        snr = float(row.get("target_snr_db", 0.0))

        # Đọc nhãn phân loại đa lớp multi-hot
        target_str = row.get("multi_hot_36", "")
        if target_str.startswith("[") and target_str.endswith("]"):
            target_list = json.loads(target_str)
            target = torch.tensor(target_list, dtype=torch.float32)
        else:
            target_idx = int(row.get("target_idx", 0))
            target = torch.zeros(36, dtype=torch.float32)
            target[target_idx] = 1.0

        target_idx = int(row.get("target_idx", int(torch.argmax(target).item())))

        return {
            "sample_id": sample_id,
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "target": target,
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
        }


class KDNoiseCacheDataset(Dataset):
    """
    Bộ nạp dữ liệu tối ưu hóa tốc độ cao sử dụng mmap (memory-mapped) cache cho sóng âm dạng npy
    và các logits được lưu trước của các mô hình giáo viên (BEATs Fold0 và BEATs EMA).
    """
    def __init__(
        self,
        base_dataset: Mixture36Dataset,
        npy_cache_path: str | Path,
        t0_logits_path: str | Path | None = None,
        tema_logits_path: str | Path | None = None,
        duration_samples: int = 64000,
    ) -> None:
        self.base = base_dataset
        self.cache_path = Path(npy_cache_path)
        self.duration_samples = duration_samples

        if self.cache_path.is_file():
            # Ánh xạ bộ nhớ mmap để nạp tức thì mà không chiếm dụng toàn bộ RAM
            self.waveforms = np.load(self.cache_path, mmap_mode="r")
        else:
            self.waveforms = None

        self.t0_logits = None
        if t0_logits_path is not None and Path(t0_logits_path).is_file():
            self.t0_logits = np.load(t0_logits_path, mmap_mode="r")

        self.tema_logits = None
        if tema_logits_path is not None and Path(tema_logits_path).is_file():
            self.tema_logits = np.load(tema_logits_path, mmap_mode="r")

    def __len__(self) -> int:
        if self.waveforms is not None:
            return len(self.waveforms)
        return len(self.base)

    def __getitem__(self, idx: int) -> AudioItem:
        if self.waveforms is not None:
            raw = self.waveforms[idx]
            wave = torch.from_numpy(np.array(raw, dtype=np.float32))
            if wave.ndim == 2 and wave.shape[0] == 1:
                wave = wave.squeeze(0)
            if wave.shape[0] < self.duration_samples:
                wave = torch.nn.functional.pad(wave, (0, self.duration_samples - wave.shape[0]))
            elif wave.shape[0] > self.duration_samples:
                wave = wave[: self.duration_samples]
        else:
            wave = torch.zeros(self.duration_samples, dtype=torch.float32)

        meta = self.base[idx] if idx < len(self.base) else {
            "sample_id": f"dummy_{idx}",
            "snr_db": torch.tensor(0.0),
            "target": torch.zeros(36),
            "target_idx": torch.tensor(0),
        }

        item: AudioItem = {
            "waveform": wave,
            "target": meta["target"],
            "target_idx": meta["target_idx"],
            "snr_db": meta["snr_db"],
            "sample_id": meta["sample_id"],
        }

        if self.t0_logits is not None:
            item["t0_logits"] = torch.from_numpy(np.array(self.t0_logits[idx], dtype=np.float32))
        else:
            item["t0_logits"] = torch.zeros(36, dtype=torch.float32)

        if self.tema_logits is not None:
            item["tema_logits"] = torch.from_numpy(np.array(self.tema_logits[idx], dtype=np.float32))
        else:
            item["tema_logits"] = torch.zeros(36, dtype=torch.float32)

        return item


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Xây dựng PyTorch DataLoader tối ưu cho quá trình huấn luyện và đánh giá."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=shuffle,
    )


# Alias tương thích ngược
NoiseCacheDataset = KDNoiseCacheDataset

