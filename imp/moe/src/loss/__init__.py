from .base import DistillationLoss, Loss, SupervisedLoss
from .cross_entropy import SoftTargetCrossEntropy
from .distillation import LogitDistillationLoss

__all__ = ["DistillationLoss", "LogitDistillationLoss", "Loss", "SoftTargetCrossEntropy", "SupervisedLoss"]
