from __future__ import annotations

import pytest
import torch

from scripts.eval_frcrn_beats import CONDITIONS, evaluate_conditions
from scripts.finetune import METRIC_NAMES
from scripts.run_kd_beat import per_snr_table

LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def _loader(batch_size: int = 36):
    batches = []
    for level in LEVELS:
        labels = torch.arange(36)
        noise = torch.zeros(batch_size, 1600)
        noise[torch.arange(batch_size), labels * 40] = 1.0  # xung tại vị trí mã hoá nhãn
        batches.append({"waveform": noise * 0.0 + torch.randn(batch_size, 1600) * 0.01, "noise": noise, "target": torch.eye(36)[labels],
                        "snr_db": torch.full((batch_size,), level)})
    return batches


def _predict(waveform: torch.Tensor) -> torch.Tensor:
    index = waveform.abs().argmax(dim=-1) // 40  # giải mã nhãn từ vị trí xung
    return torch.eye(36)[index.clamp(max=35)] * 0.9 + 0.1 / 36


def test_evaluate_conditions_reports_all_conditions_and_snr_levels() -> None:
    scores = evaluate_conditions(_loader(), lambda mixture: mixture * 0.0 + 0.0, _predict, 0.07925, torch.device("cpu"))
    assert set(scores) == set(CONDITIONS)
    table = per_snr_table(scores["noise_oracle"])
    assert list(table) == ["-5", "0", "5", "10", "15", "20", "mean"]
    assert all(set(row) == set(METRIC_NAMES) for row in table.values())
    assert scores["noise_oracle"]["macro_f1"] == pytest.approx(1.0)  # nhiễu thật chứa nhãn -> trần
    assert scores["mixture_only"]["top1_accuracy"] < 0.5  # mixture chỉ là nhiễu ngẫu nhiên nhỏ


def test_frcrn_condition_uses_separator_output() -> None:
    loader = _loader()
    scores = evaluate_conditions(loader, lambda mixture: torch.zeros_like(mixture), _predict, 0.07925, torch.device("cpu"), conditions=("frcrn_beats",))
    assert set(scores) == {"frcrn_beats"}
