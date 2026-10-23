from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.load_dataset import LABELS_FALLBACK
from scripts.run_beats_noise import AudioSet36Noise, every_n_calls, noise_input

SAMPLE_RATE = 16000


def _labels() -> list[dict[str, str]]:
    with open(LABELS_FALLBACK, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def root(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    labels = _labels()
    names36 = [row["name"] for row in labels[:-4]]
    dropped = [row["name"] for row in labels[-4:]]
    mid = {row["name"]: row["mid"] for row in labels}
    clips = {
        "single": [names36[0]],
        "other": [names36[5]],
        "multi": [names36[1], names36[2]],
        "has_dropped": [names36[3], dropped[0]],
        "held_out": [names36[4]],
    }
    (tmp_path / "metadata").mkdir()
    (tmp_path / "audio").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for ytid, clip_labels in clips.items():
        sf.write(tmp_path / "audio" / f"{ytid}.wav", rng.normal(size=6 * SAMPLE_RATE).astype(np.float32) * 0.1, SAMPLE_RATE)
        rows.append({"split": "train", "status": "copied", "ytid": ytid, "label_mids": ",".join(mid[n] for n in clip_labels)})
    with open(tmp_path / "metadata" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path, names36, dropped


def test_filters_dropped_labels_leaks_and_multilabel(root: tuple[Path, list[str], list[str]]) -> None:
    path, names36, dropped = root
    dataset = _build(path, names36, dropped, single_label=True)
    kept = sorted(p.stem for p, _ in dataset.samples)
    assert kept == ["other", "single"]
    assert dataset.class_names == names36
    targets = {p.stem: int(t.argmax()) for p, t in dataset.samples}
    assert targets == {"single": 0, "other": 5}
    assert all(t.shape == (36,) and float(t.sum()) == 1.0 for _, t in dataset.samples)


def test_keep_multilabel_takes_lowest_index(root: tuple[Path, list[str], list[str]]) -> None:
    path, names36, dropped = root
    dataset = _build(path, names36, dropped, single_label=False)
    targets = {p.stem: int(t.argmax()) for p, t in dataset.samples}
    assert targets["multi"] == 1 and "has_dropped" not in targets and "held_out" not in targets


def test_item_is_random_crop_of_requested_length(root: tuple[Path, list[str], list[str]]) -> None:
    path, names36, dropped = root
    dataset = _build(path, names36, dropped, single_label=True)
    item = dataset[0]
    assert item["waveform"].shape == (4 * SAMPLE_RATE,) and torch.equal(item["waveform"], item["noise"])
    assert item["target"].shape == (36,)


def test_noise_input_has_target_rms() -> None:
    batch = {"noise": torch.randn(3, 16000) * torch.tensor([[0.01], [0.1], [1.0]]), "target": torch.eye(36)[:3]}
    waveform, labels = noise_input(batch, 0.07925, torch.device("cpu"))
    rms = waveform.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full((3,), 0.07925), atol=1e-4)
    assert labels.tolist() == [0, 1, 2]


def _build(path: Path, names36: list[str], dropped: list[str], single_label: bool) -> AudioSet36Noise:
    return AudioSet36Noise(names36, root=path, split="train", exclude_ytids={"held_out"}, single_label=single_label, dropped=dropped)


def test_every_n_calls_uploads_on_multiples_only() -> None:
    calls: list[int] = []
    upload = every_n_calls(lambda value: calls.append(value), 3)
    for value in range(1, 8):
        upload(value)
    assert calls == [3, 6]


def test_evaluate_reports_nine_metrics_per_snr() -> None:
    from types import SimpleNamespace

    from scripts.finetune import METRIC_NAMES
    from scripts.run_beats_noise import NoiseBEATTrainer

    trainer = object.__new__(NoiseBEATTrainer)
    trainer.target_rms, trainer.device = 0.07925, torch.device("cpu")
    object.__setattr__(trainer, "_modules", {})
    trainer.__dict__["model"] = SimpleNamespace(config=SimpleNamespace(num_classes=36))
    levels = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    batches = []
    for index in range(2):
        labels = torch.arange(36)
        batches.append({
            "noise": torch.randn(36, 1600), "target": torch.eye(36)[labels],
            "snr_db": torch.full((36,), levels[index % 2 * 3]),
        })
    batches += [{"noise": torch.randn(36, 1600), "target": torch.eye(36), "snr_db": torch.full((36,), level)} for level in levels]

    def predict(waveform: torch.Tensor) -> torch.Tensor:
        return torch.eye(36).repeat(waveform.shape[0] // 36, 1) * 0.9 + 0.1 / 36

    scores = trainer._evaluate_with(predict, batches, "test")
    for level in (-5, 0, 5, 10, 15, 20):
        assert f"mAP_snr_{level}" in scores and all(f"{name}_snr_{level}" in scores for name in METRIC_NAMES)
    assert scores["macro_f1"] == pytest.approx(1.0) and scores["mAP_snr_20"] == pytest.approx(1.0)
