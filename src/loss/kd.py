from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import DistillationLoss


class KDLoss(DistillationLoss):
    """Hinton KD: T^2 * KL(softmax(teacher/T) || softmax(student/T))."""

    name = "kd"

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        return F.kl_div(
            F.log_softmax(student / T, dim=-1),
            F.softmax(teacher.detach() / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
