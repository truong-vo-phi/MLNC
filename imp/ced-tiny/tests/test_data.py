from __future__ import annotations

import csv
from pathlib import Path

import pytest
from ced_tiny.data import LabelBalancedSampler, filter_unseen_indices
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def test_label_balanced_sampler() -> None:
    labels = [0, 0, 0, 1, 1, 2]
    sampler = LabelBalancedSampler(labels, seed=42)
    order = list(sampler)
    assert len(order) == len(labels)
    assert set(order) == set(range(len(labels)))


def test_label_balanced_sampler_empty() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        LabelBalancedSampler([])


def test_filter_unseen_indices(tmp_path: Path) -> None:
    eval_csv = tmp_path / "eval.csv"
    with open(eval_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ytid_1", "start", "end"])
        writer.writerow(["ytid_2", "start", "end"])

    train_csv = tmp_path / "train.csv"
    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ytid_2", "start", "end"])
        writer.writerow(["ytid_3", "start", "end"])

    rows = [
        {"sample_id": "s1", "noise_ytid": "ytid_1"},
        {"sample_id": "s2", "noise_ytid": "ytid_2"},
        {"sample_id": "s3", "noise_ytid": "ytid_3"},
        {"sample_id": "s4", "noise_ytid": "ytid_4"},
    ]
    ds = DummyDataset(rows)
    indices = filter_unseen_indices(ds, eval_csv, [train_csv])
    assert indices == [0]
