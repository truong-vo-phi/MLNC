from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from src.loss.base import SupervisedLoss


class SoftTargetCrossEntropy(SupervisedLoss):
    name = "soft_target_cross_entropy"

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = label_smoothing

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if target.ndim == 1:
            return F.cross_entropy(prediction, target, label_smoothing=self.label_smoothing)
        if target.shape != prediction.shape:
            raise ValueError(f"target {tuple(target.shape)} and prediction {tuple(prediction.shape)} must match")
        target = target.float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.label_smoothing > 0.0:
            target = target * (1.0 - self.label_smoothing) + self.label_smoothing / prediction.shape[-1]
        return -(target * F.log_softmax(prediction, dim=-1)).sum(dim=-1).mean()
