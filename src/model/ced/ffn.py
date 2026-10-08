from __future__ import annotations

from torch import nn


class FeedForward(nn.Module):
    """Linear -> GELU -> Dropout -> Linear -> Dropout (fc1/fc2 khớp tên trong checkpoint CED)."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
