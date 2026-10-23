from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.mixdataset_speech import CleanReferenceDataset, make_mixdataset_loaders

SAMPLE_RATE = 16000


def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    (tmp_path / "labels.txt").write_text("a\nb\n", encoding="utf-8")
    for split in ("train", "validation"):
        rows = []
        for index in range(4):
            clean = rng.normal(size=4 * SAMPLE_RATE) * 0.1
            noise = rng.normal(size=4 * SAMPLE_RATE) * 0.05
            _write(tmp_path / split / "clean" / f"{index}.wav", clean)
            _write(tmp_path / split / "mixture" / f"{index}.wav", clean + noise)
            rows.append(
                {
                    "sample_id": f"{split}_{index}",
                    "mixture_path": f"{split}/mixture/{index}.wav",
                    "clean_path": f"{split}/clean/{index}.wav",
                    "noise_path": f"{split}/noise/{index}.wav",
                    "multi_hot_36": json.dumps([1, 0] if index % 2 == 0 else [0, 1]),
                    "target_snr_db": str([-5.0, 0.0][index % 2]),
                }
            )
        with open(tmp_path / split / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return tmp_path


def test_noise_is_mixture_minus_clean_and_crops_align(root: Path) -> None:
    dataset = CleanReferenceDataset(root=root, split="train", duration=1.0, random_crop=True)
    for index in range(len(dataset)):
        item = dataset[index]
        clean = sf.read(root / "train" / "clean" / f"{index}.wav", dtype="float32")[0]
        mixture = sf.read(root / "train" / "mixture" / f"{index}.wav", dtype="float32")[0]
        assert item["waveform"].shape == item["noise"].shape == (SAMPLE_RATE,)
        # noise = mixture - clean trên cùng một đoạn: tìm đoạn clean khớp với mixture - noise
        speech = (item["waveform"] - item["noise"]).numpy()
        start = int(np.argmax(np.correlate(clean, speech[:2000], mode="valid")))
        np.testing.assert_allclose(clean[start : start + SAMPLE_RATE], speech, atol=1e-6)
        np.testing.assert_allclose(mixture[start : start + SAMPLE_RATE], item["waveform"].numpy(), atol=1e-6)
    assert item["snr_db"].dtype == torch.float32 and item["target"].shape == (2,)


def test_validation_crop_is_deterministic_from_start(root: Path) -> None:
    dataset = CleanReferenceDataset(root=root, split="validation", duration=1.0)
    first, second = dataset[0], dataset[0]
    assert torch.equal(first["waveform"], second["waveform"]) and torch.equal(first["noise"], second["noise"])
    mixture = sf.read(root / "validation" / "mixture" / "0.wav", dtype="float32")[0][:SAMPLE_RATE]
    np.testing.assert_allclose(first["waveform"].numpy(), mixture, atol=1e-6)


def test_loaders_follow_config(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLNC_MIXTURE_36L", str(root))
    config = SimpleNamespace(segment_seconds=1.0, eval_segment_seconds=2.0, batch_size=2, eval_batch_size=2)
    loaders = make_mixdataset_loaders(seed=0)(config, None, 0)
    train_batch, val_batch = next(iter(loaders["train"])), next(iter(loaders["validation"]))
    assert train_batch["waveform"].shape == (2, SAMPLE_RATE) == train_batch["noise"].shape
    assert val_batch["waveform"].shape == (2, 2 * SAMPLE_RATE)


def test_train_snr_filters_train_only(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLNC_MIXTURE_36L", str(root))
    config = SimpleNamespace(segment_seconds=1.0, eval_segment_seconds=2.0, batch_size=2, eval_batch_size=2)
    loaders = make_mixdataset_loaders(seed=0, train_snr=[0])(config, None, 0)
    assert {float(x) for batch in loaders["train"] for x in batch["snr_db"]} == {0.0}
    assert {float(x) for batch in loaders["validation"] for x in batch["snr_db"]} == {-5.0, 0.0}
