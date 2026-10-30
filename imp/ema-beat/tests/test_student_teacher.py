from __future__ import annotations

import torch
from ema_beat.loss import StudentTeacherEMALoss
from ema_beat.config import EMAConfig, EMABeatTrainConfig


def test_student_teacher_loss_multilabel():
    criterion = StudentTeacherEMALoss(kd_weight=1.0, temperature=2.0)
    student_logits = torch.randn(4, 36, requires_grad=True)
    teacher_logits = torch.randn(4, 36)
    targets = torch.zeros(4, 36)
    targets[:, 0] = 1.0
    targets[:, 5] = 1.0

    loss, details = criterion(student_logits, targets, teacher_logits)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "task_loss" in details
    assert "kd_loss" in details
    assert details["kd_loss"] > 0.0

    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_student_teacher_loss_single_label():
    criterion = StudentTeacherEMALoss(kd_weight=0.5, temperature=2.0)
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    targets = torch.tensor([0, 2, 4, 1])

    loss, details = criterion(student_logits, targets, teacher_logits)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert details["kd_loss"] > 0.0

    loss.backward()
    assert student_logits.grad is not None


def test_plateau_config_defaults():
    cfg = EMABeatTrainConfig(
        plateau_factor=0.5,
        plateau_patience=2,
        min_lr=1e-6,
    )
    assert cfg.plateau_factor == 0.5
    assert cfg.plateau_patience == 2
    assert cfg.min_lr == 1e-6
