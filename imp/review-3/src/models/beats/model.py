"""
BEATs: Audio Pre-Training with Acoustic Tokenizers.
Reference: Chen et al., 'BEATs: Audio Pre-Training with Acoustic Tokenizers', ICML 2023.
Config: 12 layers, D=768, 12 heads, FFN=3072, DeepNorm scaling, patch 16x16, ~90.3M params.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseAudioClassifier


class BEATsPatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 768, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        # x: (B, 1, F, T) -> (B, embed_dim, F', T')
        x = self.proj(x)
        f_p, t_p = x.shape[2], x.shape[3]
        # Time-major order: (B, embed_dim, F', T') -> (B, T', F', embed_dim) -> (B, T'*F', embed_dim)
        x = x.permute(0, 3, 2, 1).flatten(1, 2)
        return x, f_p, t_p


class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, max_dist_f: int = 16, max_dist_t: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_dist_f = max_dist_f
        self.max_dist_t = max_dist_t
        self.bias_f = nn.Embedding(2 * max_dist_f + 1, num_heads)
        self.bias_t = nn.Embedding(2 * max_dist_t + 1, num_heads)

    def forward(self, f_p: int, t_p: int, device: torch.device) -> Tensor:
        coords_f = torch.arange(f_p, device=device)
        coords_t = torch.arange(t_p, device=device)
        diff_f = (coords_f[:, None] - coords_f[None, :] + self.max_dist_f).clamp(0, 2 * self.max_dist_f)
        diff_t = (coords_t[:, None] - coords_t[None, :] + self.max_dist_t).clamp(0, 2 * self.max_dist_t)

        b_f = self.bias_f(diff_f)  # (f, f, H)
        b_t = self.bias_t(diff_t)  # (t, t, H)
        b_total = b_t[:, None, :, None, :] + b_f[None, :, None, :, :]  # (t, f, t, f, H)
        b_total = b_total.permute(4, 0, 1, 2, 3).reshape(self.num_heads, t_p * f_p, t_p * f_p)
        return b_total.unsqueeze(0)  # (1, H, N, N)


class BEATsEncoderLayer(nn.Module):
    """DeepNorm Post-LN Transformer Layer: x = LN(alpha * x + sublayer(x))."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        total_layers: int = 12,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.alpha = math.pow(2.0 * total_layers, 0.25)  # DeepNorm alpha = (2*L)^0.25 ≈ 2.213

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, rel_pos_bias: Tensor | None = None) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(1, 2).reshape(b, n, c)

        # DeepNorm residual + LayerNorm
        x = self.attn_norm(x * self.alpha + self.drop(self.proj(attn_out)))
        x = self.ffn_norm(x * self.alpha + self.ffn(x))
        return x


class BEATs(BaseAudioClassifier):
    """
    BEATs Audio Transformer: 12 layers, D=768, 12 heads, DeepNorm, patch 16x16, ~90.3M params.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        patch_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.patch_embed = BEATsPatchEmbed(in_chans=1, embed_dim=hidden_dim, patch_size=patch_size)
        self.rel_pos_bias = RelativePositionBias2D(num_heads=num_heads)

        self.blocks = nn.ModuleList([
            BEATsEncoderLayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                total_layers=num_layers,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        tokens, f_p, t_p = self.patch_embed(x)
        bias = self.rel_pos_bias(f_p, t_p, x.device)
        for block in self.blocks:
            tokens = block(tokens, bias)
        return self.norm(tokens)

    def forward_head(self, feat: Tensor) -> Tensor:
        # Mean pooling across patch tokens -> Dropout -> Linear head
        pooled = feat.mean(dim=1)
        return self.head(self.dropout(pooled))

