from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.loss import DistillationLoss

PROB_EPS = 1e-8


class BornAgainLoss(DistillationLoss):
    name = "born_again"

    def __init__(
        self,
        kd_weight: float = 0.8,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        confidence_weighting: bool = False,
        confidence_power: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= kd_weight <= 1.0:
            raise ValueError(f"kd_weight must be in [0, 1], got {kd_weight}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if confidence_power < 0:
            raise ValueError(f"confidence_power must be >= 0, got {confidence_power}")
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.confidence_weighting = confidence_weighting
        self.confidence_power = confidence_power

    def forward(self, student: Tensor, teacher: Tensor | None, target: Tensor | None = None) -> Tensor:
        if target is None:
            raise ValueError("BornAgainLoss requires integer class targets")
        target = target.long().reshape(-1)
        student = student.float()
        if target.shape[0] != student.shape[0]:
            raise ValueError(f"student {tuple(student.shape)} and target {tuple(target.shape)} batch sizes differ")
        hard = F.cross_entropy(student, target, label_smoothing=self.label_smoothing)
        if self.kd_weight == 0.0:
            return hard
        if teacher is None:
            raise ValueError("teacher logits are required when kd_weight > 0")
        teacher = teacher.float()
        if teacher.shape != student.shape:
            raise ValueError(f"teacher {tuple(teacher.shape)} and student {tuple(student.shape)} shapes differ")

        temperature = self.temperature
        log_p_t = F.log_softmax(teacher.detach() / temperature, dim=-1)
        p_t = log_p_t.exp()
        log_p_s = F.log_softmax(student / temperature, dim=-1)
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)

        if self.confidence_weighting:
            weights = p_t.gather(1, target.unsqueeze(1)).squeeze(1).pow(self.confidence_power)
            soft = (kl * weights).sum() / weights.sum().clamp_min(PROB_EPS)
        else:
            soft = kl.mean()
        soft = soft * temperature * temperature
        return (1.0 - self.kd_weight) * hard + self.kd_weight * soft


def attention_map(hidden: Tensor) -> Tensor:
    energy = hidden.float().pow(2).mean(dim=-1)
    return F.normalize(energy, dim=-1)


def attention_transfer_terms(student_hidden: Sequence[Tensor], teacher_hidden: Sequence[Tensor]) -> Tensor:
    if len(student_hidden) != len(teacher_hidden):
        raise ValueError(f"student has {len(student_hidden)} layers, teacher has {len(teacher_hidden)}")
    losses = []
    for student, teacher in zip(student_hidden, teacher_hidden, strict=True):
        if student.shape != teacher.shape:
            raise ValueError(f"student {tuple(student.shape)} and teacher {tuple(teacher.shape)} shapes differ")
        losses.append((attention_map(student) - attention_map(teacher.detach())).pow(2).mean())
    return torch.stack(losses)


def attention_transfer_loss(student_hidden: Sequence[Tensor], teacher_hidden: Sequence[Tensor]) -> Tensor:
    return attention_transfer_terms(student_hidden, teacher_hidden).mean()
