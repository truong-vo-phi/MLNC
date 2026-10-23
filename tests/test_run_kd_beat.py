from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.run_kd_beat import DEFAULT_CONFIG, FRCRNStudent, NoiseTeacher, experiment_name, noise_dataset_class
from scripts.run_reviewkd import check_grids, read_reviewkd_config
from src.model.baseline import BaselineConfig, BaselineModel


class _Separator(nn.Module):
    """Giả FRCRNSeparator: trả nhiễu ước lượng = mixture * 0.5."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    def forward(self, waveform):
        return waveform * self.scale


def test_default_config_is_panns_cnn10_with_matching_grid() -> None:
    raw = read_reviewkd_config(DEFAULT_CONFIG)
    assert raw["model"]["arch"] == "cnn10" and raw["model"]["pretrained"] is True
    assert check_grids(raw) == (8, 24)


def test_student_feeds_noise_estimate_and_keeps_frcrn_frozen() -> None:
    student = BaselineModel(BaselineConfig(arch="cnn10", num_classes=36, freq_bins=128, num_frames=398))
    model = FRCRNStudent(_Separator(), student)
    model.train()
    assert not model.separator.training and not model.separator.scale.requires_grad
    output = model(torch.randn(2, 16000 * 4), return_features=True)
    assert output.logits.shape == (2, 36) and len(output.features) == 4


def test_noise_teacher_rejects_nonpositive_rms() -> None:
    with pytest.raises(ValueError, match="target_rms"):
        NoiseTeacher(object(), (3, 6, 9, 12), 0.0)


def test_experiment_name_encodes_flags() -> None:
    assert experiment_name("cnn10", True, None, True) == "cnn10-frcrn-kd-pre"
    assert experiment_name("cnn10", False, 0.0, False) == "cnn10-mix-ce-rand"
    assert experiment_name("cnn10", True, None, True, finetuned_teacher=True) == "cnn10-frcrn-kd-ft"
    assert experiment_name("cnn10", True, None, True, finetuned_teacher=True, logit_kd=True) == "cnn10-frcrn-kd-ft-logit"


@pytest.mark.parametrize("with_file", [True, False])
def test_noise_dataset_class_detects_noise_files(tmp_path: Path, with_file: bool) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "manifest.csv").write_text("sample_id,noise_path\na,train/noise/a.wav\n", encoding="utf-8")
    if with_file:
        (tmp_path / "train" / "noise").mkdir()
        (tmp_path / "train" / "noise" / "a.wav").write_bytes(b"x")
    assert noise_dataset_class(tmp_path) == ("file" if with_file else "residual")


def _scores(num_classes: int = 36):
    import numpy as np

    from src.train.reviewkd_trainer import snr_scores

    rng = np.random.default_rng(0)
    snrs = np.repeat([-5, 0, 5, 10, 15, 20], 72)
    labels = np.tile(np.arange(num_classes), 12)
    probs = rng.random((len(labels), num_classes))
    probs[np.arange(len(labels)), labels] += 0.8
    probs /= probs.sum(axis=1, keepdims=True)
    return snr_scores(probs, labels, snrs, num_classes)


def test_snr_scores_report_all_nine_metrics_per_level() -> None:
    from scripts.finetune import METRIC_NAMES

    scores = _scores()
    for level in (-5, 0, 5, 10, 15, 20):
        for name in METRIC_NAMES:
            assert f"{name}_snr_{level}" in scores
    for name in METRIC_NAMES:
        mean = sum(scores[f"{name}_snr_{level}"] for level in (-5, 0, 5, 10, 15, 20)) / 6
        assert scores[f"{name}_snr_mean"] == pytest.approx(mean)
    assert scores["macro_acc"] == pytest.approx(scores["balanced_accuracy_snr_mean"])  # khoá cũ vẫn khớp
    assert scores["mAP_macro"] == pytest.approx(scores["mAP_snr_mean"])


def test_per_snr_table_has_six_levels_and_mean(tmp_path: Path) -> None:
    from scripts.finetune import METRIC_NAMES
    from scripts.run_kd_beat import format_table, per_snr_table, write_per_snr_csv

    table = per_snr_table(_scores())
    assert list(table) == ["-5", "0", "5", "10", "15", "20", "mean"]
    assert all(set(row) == set(METRIC_NAMES) for row in table.values())
    path = tmp_path / "t.csv"
    write_per_snr_csv(table, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == ["snr_db", *METRIC_NAMES] and len(lines) == 8
    assert "mAP" in format_table(table).splitlines()[0]


def test_noise_teacher_can_return_grids_and_logits() -> None:
    from types import SimpleNamespace

    class _Beats(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frontend = SimpleNamespace(num_mel_bins=128)
            self.config = SimpleNamespace(input_patch_size=16)

        def freeze(self):
            return self.eval()

        def forward(self, waveform, layers=()):
            hidden = [torch.randn(waveform.shape[0], 8 * 3, 6) for _ in layers]
            return SimpleNamespace(hidden=hidden, logits=torch.randn(waveform.shape[0], 36))

    teacher = NoiseTeacher(_Beats(), (3, 6), 0.07925)
    grids = teacher(torch.randn(2, 1600))
    assert grids.shape == (2, 2, 6, 8, 3)
    grids2, logits = teacher(torch.randn(2, 1600), with_logits=True)
    assert grids2.shape == grids.shape and logits.shape == (2, 36)


def test_finetuned_teacher_does_not_download_base_beats(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace

    import scripts.run_kd_beat as module

    calls: list[str] = []
    fake = SimpleNamespace(frontend=SimpleNamespace(num_mel_bins=128), config=SimpleNamespace(input_patch_size=16))
    fake.load_pretrained = lambda path: calls.append("pretrained")
    fake.freeze = lambda: fake
    monkeypatch.setattr(module.BEATClassifier, "from_yaml", classmethod(lambda cls, path: fake))
    monkeypatch.setattr(module, "ensure_beats_checkpoint", lambda raw: calls.append("download") or tmp_path / "base.pt")
    monkeypatch.setattr(module, "load_checkpoint", lambda path, model: calls.append("finetuned"))
    monkeypatch.setattr(module, "NoiseTeacher", lambda *args, **kwargs: SimpleNamespace(to=lambda device: None))
    review = SimpleNamespace(teacher_layers=(3, 6, 9, 12), target_rms=0.07925)

    module.build_noise_teacher({"teacher": {"beats_finetuned": "x.pt"}}, review, True, torch.device("cpu"))
    assert calls == ["finetuned"]
    calls.clear()
    module.build_noise_teacher({"teacher": {"beats_finetuned": None}}, review, True, torch.device("cpu"))
    assert calls == ["download", "pretrained"]


def test_student_skips_panns_download_when_resuming(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_reviewkd as module

    raw = read_reviewkd_config(DEFAULT_CONFIG)
    downloads: list[str] = []
    monkeypatch.setattr(module, "download_checkpoint", lambda url: downloads.append(url) or "unused")
    monkeypatch.setattr(module, "resolve_url", lambda key: key)
    student = module.build_student(raw, load_pretrained=False)
    assert downloads == [] and student.config.arch == "cnn10"
