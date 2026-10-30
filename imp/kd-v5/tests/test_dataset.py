from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import Dataset

from scripts.load_dataset import (
    Mixture36Dataset,
    NoiseCacheDataset,
    SyntheticDataset,
    build_cached_loader,
    build_dataloader,
    dataset_class_names,
    subsample,
)
from src.train import DatasetConfig


class ToyDataset(Dataset):
    class_names = ["a", "b", "c"]
    rows = [{"sample_id": f"x{i}"} for i in range(3)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {
            "waveform": torch.zeros(8),
            "target": torch.tensor([1.0, 0.0, 0.0]),
            "snr_db": torch.tensor(0.0),
            "sample_id": self.rows[index]["sample_id"],
        }


def write_manifest(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "labels.txt").write_text("a\nb\nc\n", encoding="utf-8")
    rows = []
    for index in range(3):
        rows.append(
            {
                "sample_id": f"train_{index:08d}",
                "multi_hot_36": json.dumps([1.0 if i == index else 0.0 for i in range(3)]),
                "target_snr_db": str([-5.0, 0.0, 5.0][index]),
                "mixture_path": f"train/mix/{index}.wav",
                "noise_path": f"train/noise/{index}.wav",
                "clean_path": f"train/clean/{index}.wav",
                "noise_ytid": f"ytid{index}",
            }
        )
    with open(root / "train" / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_mixture_dataset_filters(tmp_path: Path) -> None:
    root = tmp_path / "mix"
    write_manifest(root)
    dataset = Mixture36Dataset(root=root, split="train")
    assert len(dataset) == 3
    assert dataset.num_classes == 3

    (root / "keep.csv").write_text("sample_id\ntrain_00000000\ntrain_00000002\n", encoding="utf-8")
    filtered = Mixture36Dataset(root=root, split="train", sample_ids_file="keep.csv")
    assert [row["sample_id"] for row in filtered.rows] == ["train_00000000", "train_00000002"]

    by_snr = Mixture36Dataset(root=root, split="train", snr_db=(0.0,))
    assert len(by_snr) == 1


def test_noise_cache_replaces_waveform(tmp_path: Path) -> None:
    base = ToyDataset()
    array = np.stack([np.full(8, index, dtype=np.float32) for index in range(3)])
    cache = tmp_path / "train.npy"
    np.save(cache, array)
    (tmp_path / "train.done").write_text("3", encoding="utf-8")
    dataset = NoiseCacheDataset(base, cache, tmp_path / "train.done")
    item = dataset[2]
    assert torch.equal(item["waveform"], torch.full((8,), 2.0))
    assert item["sample_id"] == "x2"
    assert dataset.class_names == base.class_names


def test_noise_cache_rejects_wrong_length(tmp_path: Path) -> None:
    np.save(tmp_path / "train.npy", np.zeros((2, 8), dtype=np.float32))
    (tmp_path / "train.done").write_text("3", encoding="utf-8")
    with pytest.raises(ValueError):
        NoiseCacheDataset(ToyDataset(), tmp_path / "train.npy", tmp_path / "train.done")


def test_cached_loader_filters_without_breaking_cache_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mix"
    write_manifest(root)
    for index in range(3):
        path = root / "train" / "mix" / f"{index}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(100, dtype=np.float32), 16000)

    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "train.npy", np.stack([np.full(8, index, dtype=np.float32) for index in range(3)]))
    (cache / "train.done").write_text("3", encoding="utf-8")
    (root / "keep.csv").write_text("sample_id\ntrain_00000000\ntrain_00000002\n", encoding="utf-8")

    monkeypatch.setenv("MLNC_MIXTURE_36L", str(root))
    config = DatasetConfig(duration=4.0, train_sample_ids="keep.csv")
    loader = build_cached_loader(config, "train", cache, batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))

    assert batch["sample_id"] == ["train_00000000", "train_00000002"]
    assert torch.equal(batch["waveform"][:, 0], torch.tensor([0.0, 2.0]))


def test_subsample_and_class_names(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npy"
    np.save(cache, np.zeros((3, 8), dtype=np.float32))
    base = NoiseCacheDataset(ToyDataset(), cache, None)
    subset = subsample(base, 2)
    assert len(subset) == 2
    assert dataset_class_names(subset) == base.class_names


def test_synthetic_dataset_is_deterministic() -> None:
    dataset = SyntheticDataset(length=4, num_classes=3, num_samples=100)
    first = dataset[1]
    second = dataset[1]
    assert torch.equal(first["waveform"], second["waveform"])
    assert first["target"].sum() == 1.0
    loader = build_dataloader(dataset, 2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["waveform"].shape == (2, 100)
