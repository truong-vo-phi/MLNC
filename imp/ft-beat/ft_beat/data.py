from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from scripts.load_dataset import Mixture36Dataset, build_dataloader, resolve_dataset_root, subsample

from .config import FTBeatDataConfig, resolve_path

SPLITS = ("train", "validation")
OVERLAP_KEYS = ("sample_id", "noise_ytid", "speech_id")


class CachedWaveDataset(Dataset):
    def __init__(self, base: Any, path: Path) -> None:
        self.base = base
        self.array = np.load(path, mmap_mode="r")
        self.count = min(self.array.shape[0], len(base))

    @property
    def rows(self) -> list[dict[str, str]]:
        return self.base.rows[: self.count]

    def __getattr__(self, name: str) -> Any:
        if name in ("base", "array", "count"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.base.rows[index]
        return {
            "waveform": torch.from_numpy(np.array(self.array[index], dtype=np.float32)),
            "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
            "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "noise_ytid": row.get("noise_ytid", ""),
        }


def nhat_path(directory: Path, split: str, count: int) -> Path:
    npy_file = directory / f"{split}.npy"
    if not npy_file.is_file():
        raise FileNotFoundError(f"n-hat cache not ready: {npy_file}")
    return npy_file


def dataset_rows(dataset: Dataset) -> list[dict[str, str]]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        rows = getattr(base, "rows", None)
        if rows is None:
            raise TypeError(f"{type(base).__name__} does not expose rows")
        return [rows[int(index)] for index in dataset.indices]
    rows = getattr(dataset, "rows", None)
    if rows is None:
        raise TypeError(f"{type(dataset).__name__} does not expose rows")
    return list(rows)


def class_names_of(dataset: Dataset) -> list[str]:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    names = getattr(base, "class_names", None)
    if names is None:
        raise TypeError(f"{type(base).__name__} does not expose class_names")
    return list(names)


def labels_of(dataset: Dataset) -> list[int]:
    return [int(np.argmax(json.loads(row["multi_hot_36"]))) for row in dataset_rows(dataset)]


def label_counts(dataset: Dataset) -> dict[int, int]:
    return dict(Counter(labels_of(dataset)))


def snr_counts(dataset: Dataset) -> dict[float, int]:
    return dict(Counter(float(row["target_snr_db"]) for row in dataset_rows(dataset)))


class LabelBalancedSampler(Sampler[int]):
    def __init__(self, labels: Sequence[int], seed: int = 0) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        self.labels = [int(label) for label in labels]
        self.seed = seed
        self.epoch = 0
        self._by_label: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self._by_label.setdefault(label, []).append(index)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self._by_label)
        rng.shuffle(keys)
        streams = []
        for key in keys:
            indices = list(self._by_label[key])
            rng.shuffle(indices)
            streams.append(iter(indices))
        order: list[int] = []
        while streams:
            for stream in list(streams):
                index = next(stream, None)
                if index is None:
                    streams.remove(stream)
                else:
                    order.append(index)
        self.epoch += 1
        return iter(order)

    def __len__(self) -> int:
        return len(self.labels)


def resolve_mix_dataset_root(key: str) -> Path:
    for cand in (
        Path("dataset/mix-dataset-v2"),
        Path("/marimo/dataset/mix-dataset-v2"),
        Path("dataset"),
        Path("/marimo/dataset"),
    ):
        if cand.is_dir() and (cand / "train" / "manifest.csv").is_file():
            return cand
    return resolve_dataset_root(key)


def build_datasets(config: FTBeatDataConfig) -> dict[str, Dataset]:
    root = resolve_mix_dataset_root(config.dataset_key)
    datasets: dict[str, Dataset] = {}
    for split in SPLITS:
        limit = config.max_train_samples if split == "train" else config.max_eval_samples
        dataset = Mixture36Dataset(
            root=root,
            split=split,
            duration=config.duration,
            snr_db=config.snr_levels,
        )
        if config.nhat_cache_dir is not None:
            cache = nhat_path(resolve_path(config.nhat_cache_dir), split, len(dataset))
            dataset = CachedWaveDataset(dataset, cache)
        datasets[split] = subsample(dataset, limit)

    test_npy = resolve_path(config.nhat_cache_dir) / "test.npy" if config.nhat_cache_dir is not None else None
    if (config.nhat_cache_dir is None or (test_npy and test_npy.is_file())) and (
        root / "test" / "manifest.csv"
    ).is_file():
        test_ds = Mixture36Dataset(root=root, split="test", duration=config.duration, snr_db=config.snr_levels)
        if test_npy and test_npy.is_file():
            test_ds = CachedWaveDataset(test_ds, test_npy)
        datasets["test"] = subsample(test_ds, config.max_eval_samples)
    return datasets


def build_loaders(
    datasets: Mapping[str, Dataset],
    *,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
    seed: int,
) -> dict[str, DataLoader]:
    if balanced_sampling:
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=LabelBalancedSampler(labels_of(datasets["train"]), seed=seed),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = build_dataloader(datasets["train"], batch_size, shuffle=True, num_workers=num_workers)
    if len(train_loader) == 0:
        raise ValueError("train loader has no batches; increase the dataset size or lower batch_size")
    validation_loader = build_dataloader(
        datasets["validation"], eval_batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = None
    if "test" in datasets:
        test_loader = build_dataloader(datasets["test"], eval_batch_size, shuffle=False, num_workers=num_workers)
    loaders = {"train": train_loader, "validation": validation_loader}
    if test_loader is not None:
        loaders["test"] = test_loader
    return loaders


def read_ytids_from_file(path: str | Path) -> set[str]:
    file_path = Path(path)
    if not file_path.is_file():
        return set()
    with open(file_path, encoding="utf-8") as f:
        return {line.split(",", 1)[0].strip() for line in f if line.strip() and not line.startswith("#")}


def filter_unseen_indices(
    dataset: Dataset,
    eval_segments_path: str | Path | None,
    train_segments_paths: Sequence[str | Path],
) -> list[int] | None:
    if not eval_segments_path:
        return None
    eval_ytids = read_ytids_from_file(eval_segments_path)
    if not eval_ytids:
        return None
    seen_ytids: set[str] = set()
    for path in train_segments_paths:
        seen_ytids |= read_ytids_from_file(path)

    rows = dataset_rows(dataset)
    indices = [
        i
        for i, row in enumerate(rows)
        if row.get("noise_ytid") in eval_ytids and row.get("noise_ytid") not in seen_ytids
    ]
    return indices if indices else None
