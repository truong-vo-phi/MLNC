from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
URL_CONFIG = ROOT / "config" / "url.yaml"
MIXTURE_KEY = "mixture-36l"
SAMPLE_RATE = 16000
TEACHER_CACHE_DIR = "cache"

logger = logging.getLogger(__name__)


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    snr_db: Tensor
    sample_id: str
    noise: Tensor
    clean: Tensor


def dataset_env_name(key: str) -> str:
    return f"MLNC_{key.upper().replace('-', '_')}"


def resolve_dataset_root(key: str, url_config: Path = URL_CONFIG) -> Path:
    value = os.environ.get(dataset_env_name(key))
    if not value:
        with open(url_config, encoding="utf-8") as f:
            value = yaml.safe_load(f).get(key)
    if not value:
        raise KeyError(f"'{key}' is missing or empty in {url_config}")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fit_length(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) >= num_samples:
        return waveform[:num_samples]
    return np.pad(waveform, (0, num_samples - len(waveform)))


def read_audio(
    path: Path,
    sample_rate: int,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    info = sf.info(path)
    start, frames = 0, -1
    if num_samples is not None:
        frames = round(num_samples * info.samplerate / sample_rate)
        slack = max(info.frames - frames, 0)
        start = int(rng.integers(0, slack + 1)) if rng is not None and slack > 0 else 0
    data, source_rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, source_rate // divisor)
    return waveform


def load_waveform(
    path: Path,
    sample_rate: int,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> Tensor:
    waveform = read_audio(path, sample_rate, num_samples, rng)
    if num_samples is not None:
        waveform = fit_length(waveform, num_samples)
    return torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))


class AudioDataset(Dataset, ABC):
    dataset_key: str

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = None,
    ) -> None:
        self.root = Path(root) if root else resolve_dataset_root(self.dataset_key)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = None if duration is None else int(duration * sample_rate)

    def read(self, path: Path, rng: np.random.Generator | None = None) -> Tensor:
        return load_waveform(path, self.sample_rate, self.num_samples, rng)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> AudioItem: ...


class LabeledAudioDataset(AudioDataset):
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class Mixture36Dataset(LabeledAudioDataset):
    dataset_key = MIXTURE_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = 4.0,
        snr_db: tuple[float, ...] | None = None,
        sample_ids_file: str | Path | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        with open(self.root / "labels.txt", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        rows = read_csv(self.root / split / "manifest.csv")
        if snr_db is not None:
            allowed = set(snr_db)
            rows = [row for row in rows if float(row["target_snr_db"]) in allowed]
        if sample_ids_file is not None:
            ids_path = Path(sample_ids_file)
            if not ids_path.is_absolute():
                ids_path = self.root / ids_path
            keep = {row["sample_id"] for row in read_csv(ids_path)}
            rows = [row for row in rows if row["sample_id"] in keep]
            if not rows:
                raise ValueError(f"{ids_path}: no sample_id matched the {split} manifest")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        return {
            "waveform": self.read(self.root / row["mixture_path"]),
            "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
            "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


class NoiseCacheDataset(Dataset):
    def __init__(self, base: Dataset, path: str | Path, done_path: str | Path | None = None) -> None:
        self.base = base
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"noise cache not found: {self.path}")
        self._array = np.load(self.path, mmap_mode="r")
        if len(self._array) != len(base):
            raise ValueError(f"{self.path}: {len(self._array)} rows but the dataset has {len(base)}")
        if done_path is not None:
            done = Path(done_path)
            if not done.is_file():
                raise FileNotFoundError(f"cache marker not found: {done}")
            if done.read_text().strip() != str(len(base)):
                raise ValueError(f"{done}: cache is not marked complete for {len(base)} samples")

    @property
    def class_names(self) -> list[str]:
        return self.base.class_names

    @property
    def rows(self) -> list[dict[str, str]]:
        return self.base.rows

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> AudioItem:
        if hasattr(self.base, "rows") and "multi_hot_36" in self.base.rows[index]:
            row = self.base.rows[index]
            return {
                "waveform": torch.from_numpy(np.array(self._array[index], dtype=np.float32)),
                "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
                "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
                "sample_id": row["sample_id"],
            }
        item = dict(self.base[index])
        item["waveform"] = torch.from_numpy(np.array(self._array[index], dtype=np.float32))
        return item



class SyntheticDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        num_classes: int = 36,
        num_samples: int = SAMPLE_RATE * 4,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.seed = seed
        self.snr_levels = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> AudioItem:
        generator = torch.Generator().manual_seed(self.seed + index)
        waveform = torch.randn(self.num_samples, generator=generator) * 0.1
        target = torch.zeros(self.num_classes)
        target[index % self.num_classes] = 1.0
        return {
            "waveform": waveform,
            "target": target,
            "snr_db": torch.tensor(self.snr_levels[index % len(self.snr_levels)], dtype=torch.float32),
            "sample_id": f"synthetic_{index:08d}",
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def subsample(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    indices = np.linspace(0, len(dataset) - 1, max_samples).round().astype(int)
    return Subset(dataset, indices.tolist())


@dataclass(frozen=True)
class CachedLoaders:
    train: DataLoader
    validation: DataLoader
    class_names: list[str]


def sample_ids_path(config: Any, split: str) -> str | None:
    return {
        "train": config.train_sample_ids,
        "validation": config.validation_sample_ids,
        "test": config.test_sample_ids,
    }[split]


def resolve_sample_ids_file(root: Path, sample_ids_file: str | Path) -> Path:
    path = Path(sample_ids_file)
    return path if path.is_absolute() else root / path


def selection_indices(
    rows: list[dict[str, str]],
    snr_db: tuple[float, ...] | None,
    sample_ids_file: str | Path | None,
    root: Path,
) -> list[int]:
    indices = list(range(len(rows)))
    if snr_db is not None:
        allowed = set(snr_db)
        indices = [index for index in indices if float(rows[index]["target_snr_db"]) in allowed]
    if sample_ids_file is not None:
        keep = {row["sample_id"] for row in read_csv(resolve_sample_ids_file(root, sample_ids_file))}
        indices = [index for index in indices if rows[index]["sample_id"] in keep]
    if not indices:
        raise ValueError("no sample matched the requested sample_ids_file / snr_db filter")
    return indices


def build_cached_loader(
    config: Any,
    split: str,
    cache_dir: str | Path,
    batch_size: int,
    max_samples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = False,
    sample_ids_file: str | Path | None = None,
) -> DataLoader:
    base = Mixture36Dataset(split=split, duration=config.duration)
    cache = Path(cache_dir)
    dataset: Dataset = NoiseCacheDataset(base, cache / f"{split}.npy", cache / f"{split}.done")
    ids_file = sample_ids_file if sample_ids_file is not None else sample_ids_path(config, split)
    indices = selection_indices(base.rows, config.snr_db, ids_file, base.root)
    if len(indices) != len(base):
        dataset = Subset(dataset, indices)
    return build_dataloader(subsample(dataset, max_samples), batch_size, shuffle, num_workers)


def dataset_class_names(dataset: Dataset) -> list[str]:
    current: Dataset = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current.class_names


def build_cached_loaders(
    config: Any,
    cache_dir: str | Path,
    train_batch_size: int,
    eval_batch_size: int,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> CachedLoaders:
    train = build_cached_loader(config, "train", cache_dir, train_batch_size, max_samples, num_workers, shuffle=True)
    validation = build_cached_loader(
        config, "validation", cache_dir, eval_batch_size, max_samples, num_workers, shuffle=False
    )
    return CachedLoaders(train=train, validation=validation, class_names=dataset_class_names(train.dataset))
