from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import DistillationLoss


class CRDLoss(DistillationLoss):
    """Contrastive Representation Distillation ở mức patch (trước pooling).

    Student/teacher patch features (B, N, D) được chiếu về `proj_dim`, L2-normalize. Với mỗi vị trí
    patch n, cặp (student_b, teacher_b) là positive, các teacher_{b'} khác trong batch là negative (InfoNCE).
    """

    name = "crd"

    def __init__(self, student_dim: int = 768, teacher_dim: int = 768, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.proj_s = nn.Linear(student_dim, proj_dim)
        self.proj_t = nn.Linear(teacher_dim, proj_dim)
        self.temperature = temperature

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = F.normalize(self.proj_s(student), dim=-1)  # (B, N, P)
        t = F.normalize(self.proj_t(teacher.detach()), dim=-1)
        sim = torch.einsum("bnp,cnp->nbc", s, t) / self.temperature  # (N, B, B)
        n, b, _ = sim.shape
        target = torch.arange(b, device=sim.device).repeat(n)
        return F.cross_entropy(sim.reshape(n * b, b), target)
