from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ban_beat.nhat import CachedWaveDataset, nhat_path


class FakeBase:
    def __init__(self, count: int) -> None:
        self.class_names = ["a", "b", "c"]
        self.rows = [
            {
                "sample_id": f"s{i}",
                "multi_hot_36": json.dumps([1 if j == i % 3 else 0 for j in range(3)]),
                "target_snr_db": str(float(i)),
            }
            for i in range(count)
        ]

    def __len__(self) -> int:
        return len(self.rows)


def test_cached_wave_dataset_returns_cached_waveform_and_labels(tmp_path: Path) -> None:
    array = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    path = tmp_path / "train.npy"
    np.save(path, array)
    dataset = CachedWaveDataset(FakeBase(4), path)
    item = dataset[2]
    torch.testing.assert_close(item["waveform"], torch.from_numpy(array[2]))
    assert item["target"].tolist() == [0.0, 0.0, 1.0]
    assert float(item["snr_db"]) == 2.0
    assert item["sample_id"] == "s2"
    assert len(dataset) == 4
    assert dataset.class_names == ["a", "b", "c"]
    assert len(dataset.rows) == 4


def test_cached_wave_dataset_rejects_row_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "train.npy"
    np.save(path, np.zeros((3, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="rows"):
        CachedWaveDataset(FakeBase(4), path)


def test_nhat_path_requires_done_and_matching_count(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        nhat_path(tmp_path, "train", 4)
    (tmp_path / "train.done").write_text("3", encoding="utf-8")
    with pytest.raises(ValueError, match="clip"):
        nhat_path(tmp_path, "train", 4)
    (tmp_path / "train.done").write_text("4", encoding="utf-8")
    assert nhat_path(tmp_path, "train", 4) == tmp_path / "train.npy"
