from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from ban_beat.loss import BornAgainLoss, attention_transfer_loss, attention_transfer_terms

CLASSES = 5
BATCH = 4


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    student = torch.randn(BATCH, CLASSES, generator=generator)
    teacher = torch.randn(BATCH, CLASSES, generator=generator)
    target = torch.tensor([0, 1, 2, 3])
    return student, teacher, target


def test_kd_weight_zero_is_cross_entropy() -> None:
    student, _, target = _inputs()
    loss = BornAgainLoss(kd_weight=0.0)
    torch.testing.assert_close(loss(student, None, target), F.cross_entropy(student, target))


def test_temperature_scaling_uses_t_squared() -> None:
    student, teacher, target = _inputs()
    temperature = 2.5
    value = BornAgainLoss(kd_weight=1.0, temperature=temperature)(student, teacher, target)
    log_p_t = F.log_softmax(teacher / temperature, dim=-1)
    log_p_s = F.log_softmax(student / temperature, dim=-1)
    expected = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1).mean() * temperature**2
    torch.testing.assert_close(value, expected)


def test_teacher_is_detached() -> None:
    student, teacher, target = _inputs()
    student = student.requires_grad_(True)
    teacher = teacher.requires_grad_(True)
    value = BornAgainLoss(kd_weight=1.0)(student, teacher, target)
    value.backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_confidence_weighting_is_finite() -> None:
    student, teacher, target = _inputs()
    loss = BornAgainLoss(kd_weight=1.0, confidence_weighting=True, confidence_power=2.0)
    value = loss(student, teacher, target)
    assert torch.isfinite(value)


def test_missing_teacher_raises() -> None:
    student, _, target = _inputs()
    with pytest.raises(ValueError, match="teacher logits"):
        BornAgainLoss(kd_weight=0.5)(student, None, target)


def test_shape_mismatch_raises() -> None:
    student, teacher, target = _inputs()
    with pytest.raises(ValueError, match="shapes differ"):
        BornAgainLoss()(student, teacher[:, :3], target)


def test_missing_target_raises() -> None:
    student, teacher, _ = _inputs()
    with pytest.raises(ValueError, match="integer class targets"):
        BornAgainLoss()(student, teacher)


def test_attention_transfer_loss_is_positive_for_different_hidden() -> None:
    generator = torch.Generator().manual_seed(1)
    student = (torch.randn(2, 6, 8, generator=generator),)
    teacher = (torch.randn(2, 6, 8, generator=generator),)
    assert attention_transfer_loss(student, teacher).item() > 0.0


def test_attention_transfer_detaches_teacher() -> None:
    student = (torch.randn(2, 6, 8, requires_grad=True),)
    teacher = (torch.randn(2, 6, 8, requires_grad=True),)
    attention_transfer_loss(student, teacher).backward()
    assert teacher[0].grad is None
    assert student[0].grad is not None


def test_attention_transfer_rejects_layer_count_mismatch() -> None:
    hidden = (torch.randn(2, 6, 8),)
    with pytest.raises(ValueError, match="layers"):
        attention_transfer_loss(hidden, hidden + hidden)


def test_attention_transfer_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        attention_transfer_loss((torch.randn(2, 6, 8),), (torch.randn(2, 5, 8),))


def test_attention_transfer_loss_is_mean_over_layers() -> None:
    generator = torch.Generator().manual_seed(2)
    student = tuple(torch.randn(2, 6, 8, generator=generator) for _ in range(3))
    teacher = tuple(torch.randn(2, 6, 8, generator=generator) for _ in range(3))
    terms = attention_transfer_terms(student, teacher)
    assert terms.shape == (3,)
    torch.testing.assert_close(attention_transfer_loss(student, teacher), terms.mean())
