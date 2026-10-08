from __future__ import annotations

import torch
from torch import nn


class ClassifierHead(nn.Module):
    """Mean pooling theo trục patch -> LayerNorm -> Linear."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(patch_features.mean(dim=1)))
