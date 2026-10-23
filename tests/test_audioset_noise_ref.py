import csv
import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from scripts.audioset_noise_ref import (
    AudioSetNoiseMixDataset,
    build_clip_cache,
    clip_cache_path,
    compute_offsets,
    locate_segment,
    summarize_errors,
)

SR = 16000
LENGTH = 4 * SR
MANIFEST_FIELDS = [
    "sample_id",
    "split",
    "clean_path",
    "noise_path",
    "mixture_path",
    "noise_ytid",
    "multi_hot_36",
    "target_snr_db",
    "noise_scale",
    "post_gain",
]


def write_wav(path, data, rate=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), rate, subtype="FLOAT")


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    audioset = tmp_path / "AudioSet"
    source = rng.normal(0, 0.1, 6 * 44100)
    write_wav(audioset / "audio" / "yt1.wav", source, 44100)
    reference_clip = resample_poly(source.astype(np.float32), 160, 441)  # cùng đường xử lý với read_audio

    root = tmp_path / "mix"
    (root).mkdir()
    (root / "labels.txt").write_text("a\nb\n", encoding="utf-8")
    rows = []
    # (offset, scale, gain, noise lấy từ clip hay là nhiễu không liên quan)
    for i, (offset, scale, gain, related) in enumerate([(5000, 0.5, 1.0, True), (30000, 2.0, 0.8, True), (1000, 1.0, 1.0, False)]):
        sid = f"train_{i:05d}"
        noise = (
            gain * scale * reference_clip[offset : offset + LENGTH]
            if related
            else rng.normal(0, 0.1, LENGTH).astype(np.float32)
        )
        clean = rng.normal(0, 0.1, LENGTH).astype(np.float32)
        write_wav(root / "train" / "noise" / f"{sid}.wav", noise)
        write_wav(root / "train" / "clean" / f"{sid}.wav", clean)
        write_wav(root / "train" / "mixture" / f"{sid}.wav", clean + noise)
        rows.append(
            {
                "sample_id": sid,
                "split": "train",
                "clean_path": f"train/clean/{sid}.wav",
                "noise_path": f"train/noise/{sid}.wav",
                "mixture_path": f"train/mixture/{sid}.wav",
                "noise_ytid": "yt1",
                "multi_hot_36": json.dumps([1, 0]),
                "target_snr_db": "0.0",
                "noise_scale": str(scale),
                "post_gain": str(gain),
            }
        )
    with open(root / "train" / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"root": root, "audioset": audioset, "cache": tmp_path / "cache", "out": tmp_path / "offsets_train.csv", "clip": reference_clip}


def test_locate_segment_recovers_planted_offset():
    rng = np.random.default_rng(1)
    clip = rng.normal(0, 0.1, 10 * SR).astype(np.float32)
    reference = 0.7 * clip[12345 : 12345 + LENGTH]
    offset, error = locate_segment(clip, reference, 0.7)
    assert offset == 12345
    assert error < 1e-10


def test_locate_segment_reports_large_error_for_wrong_gain():
    rng = np.random.default_rng(2)
    clip = rng.normal(0, 0.1, 10 * SR).astype(np.float32)
    reference = clip[2000 : 2000 + LENGTH]
    _, error = locate_segment(clip, reference, 0.5)
    assert error == pytest.approx(0.25, rel=1e-3)


def test_clip_cache_matches_loader_pipeline(world):
    assert build_clip_cache(["yt1"], world["audioset"], world["cache"], workers=1) == 1
    cached = np.load(clip_cache_path(world["cache"], "yt1"))
    assert cached.dtype == np.float32
    np.testing.assert_allclose(cached, world["clip"], atol=1e-6)


def test_compute_offsets_and_dataset_use_audioset_reference(world):
    compute_offsets("train", world["out"], world["cache"], root=world["root"], audioset_root=world["audioset"], workers=1)
    with open(world["out"], encoding="utf-8", newline="") as f:
        table = {row["sample_id"]: row for row in csv.DictReader(f)}
    assert abs(int(table["train_00000"]["offset"]) - 5000) <= 1
    assert abs(int(table["train_00001"]["offset"]) - 30000) <= 1
    assert float(table["train_00000"]["error"]) < 1e-6
    assert float(table["train_00002"]["error"]) > 0.5  # nhiễu không liên quan tới clip
    summary = summarize_errors(world["out"])
    assert summary["rows"] == 3 and summary["kept@0.01"] == pytest.approx(2 / 3)

    dataset = AudioSetNoiseMixDataset(world["out"], world["cache"], max_error=0.01, root=world["root"], split="train", duration=2.0)
    assert len(dataset) == 2 and dataset.total_rows == 3
    item = dataset[1]
    stored, _ = sf.read(world["root"] / "train" / "noise" / "train_00001.wav", dtype="float32")
    assert item["noise"].shape == (2 * SR,) and item["waveform"].shape == (2 * SR,)
    np.testing.assert_allclose(item["noise"].numpy(), stored[: 2 * SR], atol=1e-5)


def test_dataset_never_reads_noise_path(world):
    compute_offsets("train", world["out"], world["cache"], root=world["root"], audioset_root=world["audioset"], workers=1)
    for path in (world["root"] / "train" / "noise").glob("*.wav"):
        path.unlink()
    dataset = AudioSetNoiseMixDataset(world["out"], world["cache"], max_error=0.01, root=world["root"], split="train", duration=4.0)
    assert dataset[0]["noise"].shape == (LENGTH,)
