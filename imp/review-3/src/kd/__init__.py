"""
Knowledge Distillation Package for Review-3 (Task 3).
"""

from src.kd.losses import (
    CompositeKDLoss,
    CRDLoss,
    LogitKDLoss,
    MGDLoss,
)
from src.kd.trainer import KDTrainer

__all__ = [
    "LogitKDLoss",
    "CRDLoss",
    "MGDLoss",
    "CompositeKDLoss",
    "KDTrainer",
]

