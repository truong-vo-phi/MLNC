from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import DistillationLoss


class HierarchicalContextLoss(DistillationLoss):
    name: ClassVar[str] = "hcl"

    def __init__(self, levels: Sequence[int] = (4, 2, 1)) -> None:
        super().__init__()
        self.levels = tuple(levels)

    def per_level(self, student: Tensor, teacher: Tensor) -> Tensor:
        if student.shape != teacher.shape or student.dim() != 5:
            raise ValueError(f"expected matching (L, B, C, H, W) tensors, got {student.shape} and {teacher.shape}")
        values = []
        for fs, ft in zip(student, teacher, strict=True):
            loss = F.mse_loss(fs, ft)
            weight, total = 1.0, 1.0
            for level in self.levels:
                if level >= fs.shape[-2]:
                    continue
                weight /= 2.0
                pooled = F.mse_loss(
                    F.adaptive_avg_pool2d(fs, (level, level)), F.adaptive_avg_pool2d(ft, (level, level))
                )
                loss = loss + pooled * weight
                total += weight
            values.append(loss / total)
        return torch.stack(values)

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        return self.per_level(student, teacher).sum()
