from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.pipeline import ChampionPipeline, softmax


class DummyModel(nn.Module):
    def __init__(self, val: float, num_classes: int = 36) -> None:
        super().__init__()
        self.val = val
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        b = x.shape[0]
        logits = torch.ones(b, self.num_classes) * self.val
        return SimpleNamespace(logits=logits)


def test_softmax() -> None:
    logits = np.array([[2.0, 1.0, 0.0]])
    probs = softmax(logits, temp=1.0)
    assert probs.shape == (1, 3)
    assert np.isclose(np.sum(probs), 1.0)

    probs_high_temp = softmax(logits, temp=10.0)
    assert probs_high_temp[0, 0] < probs[0, 0]


def test_champion_pipeline_routing_and_weights() -> None:
    device = torch.device("cpu")
    low_student = DummyModel(val=1.0)
    high_student = DummyModel(val=2.0)
    teacher_fold0 = DummyModel(val=3.0)
    teacher_ema = DummyModel(val=4.0)

    pipe = ChampionPipeline(
        low_student=low_student,
        high_student=high_student,
        teacher_fold0=teacher_fold0,
        teacher_ema=teacher_ema,
        device=device,
        snr_threshold_db=18.0,
        temp_student=1.0,
        temp_teacher=1.0,
        low_weights=(0.30, 0.45, 0.25),
        high_weights=(0.35, 0.40, 0.25),
    )

    wave = torch.randn(4, 16000)
    snrs = np.array([-5.0, 10.0, 18.0, 20.0])

    probs = pipe.predict_batch(wave, snrs)
    assert probs.shape == (4, 36)
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(4), atol=1e-5)

    preds = pipe.predict(wave, snrs)
    assert preds.shape == (4,)
