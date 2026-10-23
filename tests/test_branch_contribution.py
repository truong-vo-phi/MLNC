from __future__ import annotations

import numpy as np
import pytest

from scripts.branch_contribution import analyze, format_table

CLASSES = 6
DIM = 32


def _split(rng: np.random.Generator, per_snr: int, noise_signal: dict[int, float], speech_signal: dict[int, float]):
    prototypes_n = rng.normal(size=(CLASSES, DIM))
    prototypes_s = rng.normal(size=(CLASSES, DIM))
    labels, snrs, noise, speech = [], [], [], []
    for snr in noise_signal:
        for _ in range(per_snr):
            label = int(rng.integers(CLASSES))
            labels.append(label)
            snrs.append(snr)
            noise.append(noise_signal[snr] * prototypes_n[label] + rng.normal(size=DIM))
            speech.append(speech_signal[snr] * prototypes_s[label] + rng.normal(size=DIM))
    return {
        "noise_embedding": np.array(noise),
        "speech_embedding": np.array(speech),
        "label": np.array(labels),
        "snr_db": np.array(snrs, dtype=float),
    }


def test_noise_share_follows_signal_strength() -> None:
    rng = np.random.default_rng(0)
    strength_noise = {-5: 2.0, 20: 0.1}  # nhánh noise mạnh ở SNR thấp, yếu ở SNR cao
    strength_speech = {-5: 0.1, 20: 2.0}
    train = _split(rng, 600, strength_noise, strength_speech)
    test = _split(np.random.default_rng(0), 600, strength_noise, strength_speech)
    report = analyze(train, test, repeats=10, bootstrap=50, num_classes=CLASSES)
    assert report["-5"]["shapley_noise_pct"] > 80
    assert report["20"]["shapley_noise_pct"] < 20
    assert report["-5"]["ablation_noise_pct"] > report["20"]["ablation_noise_pct"]


def test_percentages_sum_and_chance_floor() -> None:
    rng = np.random.default_rng(1)
    strength = {0: 1.0, 10: 1.0}
    train = _split(rng, 400, strength, strength)
    test = _split(np.random.default_rng(1), 400, strength, strength)
    report = analyze(train, test, repeats=10, bootstrap=20, num_classes=CLASSES)
    for row in report.values():
        assert row["shapley_noise_pct"] + row["shapley_speech_pct"] == pytest.approx(100.0, abs=1e-6)
        assert row["ablation_noise_pct"] + row["ablation_speech_pct"] == pytest.approx(100.0, abs=1e-6)
        assert row["acc_none"] == pytest.approx(1 / CLASSES, abs=0.06)
        assert row["acc_full"] >= max(row["acc_no_noise"], row["acc_no_speech"])
    assert "shap N%" in format_table(report)
