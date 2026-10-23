from __future__ import annotations

import pytest
import torch

from scripts.eval_frcrn_snr import frcrn_snr_table

LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def _loader():
    torch.manual_seed(0)
    batches = []
    for level in LEVELS:
        speech, noise = torch.randn(4, 1600), torch.randn(4, 1600)
        batches.append({"waveform": speech + noise, "noise": noise, "snr_db": torch.full((4,), level), "_speech": speech})
    return batches


def test_perfect_noise_estimate_gives_high_si_snr_and_all_levels() -> None:
    loader = _loader()
    index = iter(range(len(loader)))
    # tách nhiễu hoàn hảo: ước lượng = nhiễu thật của batch đang xét
    scores = frcrn_snr_table(loader, lambda mixture: loader[next(index)]["noise"], torch.device("cpu"))
    for level in (-5, 0, 5, 10, 15, 20):
        assert scores[f"noise_si_snr_{level}"] > 60 and f"speech_si_snr_{level}" in scores
    assert scores["noise_si_snr_hi"] == pytest.approx((scores["noise_si_snr_15"] + scores["noise_si_snr_20"]) / 2)


def test_zero_estimate_is_much_worse_than_perfect() -> None:
    scores = frcrn_snr_table(_loader(), lambda mixture: mixture * 0.5, torch.device("cpu"))
    assert scores["noise_si_snr"] < 30
