from __future__ import annotations

import torch
from torch import nn

from .attention import Attention
from .config import CEDConfig
from .ffn import FeedForward


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: CEDConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = Attention(
            config.embed_dim, config.num_heads, config.qkv_bias, config.attn_drop_rate, config.drop_rate
        )
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp = FeedForward(config.embed_dim, int(config.embed_dim * config.mlp_ratio), config.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
