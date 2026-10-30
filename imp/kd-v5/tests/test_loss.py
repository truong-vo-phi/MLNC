from __future__ import annotations

import pytest
import torch

from src.loss import LogitDistillationLoss, SoftTargetCrossEntropy


def test_soft_target_ce_one_hot_matches_cross_entropy() -> None:
    logits = torch.tensor([[4.0, 0.0, -1.0], [0.0, 3.0, 1.0]])
    target = torch.tensor([0, 1])
    expected = torch.nn.functional.cross_entropy(logits, target)
    actual = SoftTargetCrossEntropy()(logits, target)
    assert torch.allclose(actual, expected)


def test_soft_target_ce_is_finite_with_mixup() -> None:
    logits = torch.randn(4, 6)
    first = torch.eye(6)[torch.tensor([0, 1, 2, 3])]
    target = 0.7 * first + 0.3 * first.roll(1, dims=0)
    loss = SoftTargetCrossEntropy(label_smoothing=0.1)(logits, target)
    assert torch.isfinite(loss)


def test_logit_distillation_zero_when_equal() -> None:
    logits = torch.randn(5, 7)
    loss = LogitDistillationLoss(temperature=4.0)(logits, logits)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_logit_distillation_positive_when_different() -> None:
    student = torch.randn(5, 7)
    teacher = torch.randn(5, 7)
    loss = LogitDistillationLoss(temperature=4.0)(student, teacher)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        LogitDistillationLoss()(torch.randn(2, 3), torch.randn(2, 4))
