from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedWaveDataset(Dataset):
    def __init__(self, base: Any, path: Path) -> None:
        self.base = base
        self.array = np.load(path, mmap_mode="r")
        if self.array.shape[0] != len(base):
            raise ValueError(f"{path}: {self.array.shape[0]} rows, dataset has {len(base)}")

    @property
    def rows(self) -> list[dict[str, str]]:
        return self.base.rows

    def __getattr__(self, name: str) -> Any:
        if name in ("base", "array"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.base.rows[index]
        return {
            "waveform": torch.from_numpy(np.array(self.array[index], dtype=np.float32)),
            "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
            "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


def nhat_path(directory: Path, split: str, count: int) -> Path:
    done = directory / f"{split}.done"
    if not done.is_file():
        raise FileNotFoundError(f"n-hat cache chưa sẵn sàng: {done}")
    written = int(done.read_text(encoding="utf-8").strip())
    if written != count:
        raise ValueError(f"{done}: {written} clip, dataset có {count}")
    return directory / f"{split}.npy"
