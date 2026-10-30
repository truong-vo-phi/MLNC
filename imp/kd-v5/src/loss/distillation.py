from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .base import DistillationLoss


class LogitDistillationLoss(DistillationLoss):
    name = "logit_distillation"

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        if student.shape != teacher.shape:
            raise ValueError(f"student {tuple(student.shape)} and teacher {tuple(teacher.shape)} must match")
        temperature = self.temperature
        return (
            F.kl_div(
                F.log_softmax(student / temperature, dim=-1),
                F.softmax(teacher / temperature, dim=-1),
                reduction="batchmean",
            )
            * temperature**2
        )
