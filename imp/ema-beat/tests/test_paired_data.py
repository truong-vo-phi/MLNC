from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile
import torch

from ema_beat.data import (
    NUM_CLASSES,
    NUM_SAMPLES,
    CacheNhatDataset,
    PairedStudentTeacherDataset,
    find_mix_manifest,
)


def _make_world(tmp_path: Path, rows: int = 4, with_audio: bool = True) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    cache = dataset / "cache"
    cache.mkdir(parents=True)
    np.save(cache / "train.npy", np.random.randn(rows, NUM_SAMPLES).astype(np.float32))

    mix = dataset / "mix-dataset-v2" / "mix-dataset-v2" / "train"
    mix.mkdir(parents=True)
    with open(mix / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "noise_ytid", "multi_hot_36", "target_snr_db"])
        writer.writeheader()
        for i in range(rows):
            hot = [0] * NUM_CLASSES
            hot[i % NUM_CLASSES] = 1
            writer.writerow(
                {
                    "sample_id": f"train_{i}",
                    "split": "train",
                    "noise_ytid": f"yt{i % 2}",
                    "multi_hot_36": json.dumps(hot),
                    "target_snr_db": 5.0,
                }
            )

    audioset = tmp_path / "AudioSet"
    (audioset / "audio").mkdir(parents=True)
    if with_audio:
        for k in range(2):
            wave = np.full(NUM_SAMPLES, 0.1 * (k + 1), dtype=np.float32)
            wavfile.write(audioset / "audio" / f"yt{k}.wav", 16000, wave)
    return cache, audioset


def test_find_manifest_by_split(tmp_path: Path) -> None:
    cache, _ = _make_world(tmp_path)
    found = find_mix_manifest(cache, "train")
    assert found is not None and found.name == "manifest.csv"
    assert find_mix_manifest(cache, "validation") is None


def test_cache_labels_not_zero(tmp_path: Path) -> None:
    cache, _ = _make_world(tmp_path)
    ds = CacheNhatDataset(cache, "train")
    assert len(ds) == 4
    assert all(ds[i]["target"].sum() == 1 for i in range(4))
    assert ds[2]["target"].argmax() == 2


def test_missing_manifest_raises(tmp_path: Path) -> None:
    cache = tmp_path / "dataset" / "cache"
    cache.mkdir(parents=True)
    np.save(cache / "train.npy", np.zeros((2, NUM_SAMPLES), dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        CacheNhatDataset(cache, "train")


def test_paired_teacher_gets_clean_audioset(tmp_path: Path) -> None:
    cache, audioset = _make_world(tmp_path)
    ds = PairedStudentTeacherDataset(cache, audioset, "train")
    assert len(ds) == 4
    item = ds[1]
    assert item["waveform"].shape == (NUM_SAMPLES,)
    assert item["teacher_waveform"].shape == (NUM_SAMPLES,)
    assert item["noise_ytid"] == "yt1"
    assert torch.allclose(item["teacher_waveform"], torch.full((NUM_SAMPLES,), 0.2), atol=1e-3)
    assert not torch.allclose(item["waveform"], item["teacher_waveform"])


def test_paired_without_audio_raises(tmp_path: Path) -> None:
    cache, audioset = _make_world(tmp_path, with_audio=False)
    with pytest.raises(FileNotFoundError):
        PairedStudentTeacherDataset(cache, audioset, "train")
