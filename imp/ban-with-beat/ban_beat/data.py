from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from scripts.load_dataset import Mixture36Dataset, build_dataloader, resolve_dataset_root, subsample

from .config import BANDataConfig, resolve_path
from .nhat import CachedWaveDataset, nhat_path

SPLITS = ("train", "validation")
OVERLAP_KEYS = ("sample_id", "noise_ytid", "speech_id")


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


def build_datasets(config: BANDataConfig) -> dict[str, Dataset]:
    root = resolve_dataset_root(config.dataset_key)
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
    return {"train": train_loader, "validation": validation_loader}


def split_overlap(datasets: Mapping[str, Dataset], keys: Sequence[str] = OVERLAP_KEYS) -> dict[str, int]:
    result: dict[str, int] = {}
    splits = list(datasets)
    for position, left in enumerate(splits):
        left_rows = dataset_rows(datasets[left])
        for right in splits[position + 1 :]:
            right_rows = dataset_rows(datasets[right])
            for key in keys:
                left_values = {row[key] for row in left_rows}
                right_values = {row[key] for row in right_rows}
                result[f"{left}:{right}:{key}"] = len(left_values & right_values)
    return result


def check_split_disjoint(datasets: Mapping[str, Dataset]) -> dict[str, int]:
    overlaps = split_overlap(datasets)
    leaked = {key: value for key, value in overlaps.items() if value > 0}
    if leaked:
        raise ValueError(f"split leakage detected: {leaked}")
    return overlaps


def coverage_report(datasets: Mapping[str, Dataset], snr_levels: Sequence[float]) -> dict[str, Any]:
    names = class_names_of(datasets["train"])
    class_counts = {split: label_counts(dataset) for split, dataset in datasets.items()}
    levels = {split: snr_counts(dataset) for split, dataset in datasets.items()}
    return {
        "class_counts": {
            split: {name: counts.get(index, 0) for index, name in enumerate(names)}
            for split, counts in class_counts.items()
        },
        "snr_counts": {
            split: {f"{level:g}": levels[split].get(float(level), 0) for level in snr_levels} for split in levels
        },
        "missing_classes": {
            split: [name for index, name in enumerate(names) if counts.get(index, 0) == 0]
            for split, counts in class_counts.items()
        },
    }
