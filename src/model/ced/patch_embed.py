from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """spectrogram (B, 1, F, T) -> patch grid (B, C, F/p, T/p) bằng Conv2d stride = patch."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
