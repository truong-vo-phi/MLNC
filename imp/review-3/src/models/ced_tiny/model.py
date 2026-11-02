"""
CED-Tiny: Consistent Ensemble Distillation Tiny Audio Transformer.
Reference: Dinkel et al., 'CED: Consistent Ensemble Distillation for Audio Tagging', ICASSP 2024.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseAudioClassifier


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        # x: (B, 1, F, T) -> (B, embed_dim, F', T')
        x = self.proj(x)
        f_p, t_p = x.shape[2], x.shape[3]
        # Time-major ordering: (B, embed_dim, F', T') -> (B, T', F', embed_dim) -> (B, T'*F', embed_dim)
        x = x.permute(0, 3, 2, 1).flatten(1, 2)
        return x, f_p, t_p


class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, max_dist_f: int = 16, max_dist_t: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_dist_f = max_dist_f
        self.max_dist_t = max_dist_t
        # Tables for freq and time relative offsets
        self.bias_f = nn.Embedding(2 * max_dist_f + 1, num_heads)
        self.bias_t = nn.Embedding(2 * max_dist_t + 1, num_heads)

    def forward(self, f_p: int, t_p: int, device: torch.device) -> Tensor:
        # Create coordinate grids
        coords_f = torch.arange(f_p, device=device)
        coords_t = torch.arange(t_p, device=device)
        # Relative differences: (f1 - f2)
        diff_f = coords_f[:, None] - coords_f[None, :] + self.max_dist_f
        diff_f = diff_f.clamp(0, 2 * self.max_dist_f)
        diff_t = coords_t[:, None] - coords_t[None, :] + self.max_dist_t
        diff_t = diff_t.clamp(0, 2 * self.max_dist_t)

        # Lookups: (f_p, f_p, H) and (t_p, t_p, H)
        b_f = self.bias_f(diff_f)  # (f, f, H)
        b_t = self.bias_t(diff_t)  # (t, t, H)

        # Composite relative bias for time-major flattened tokens (T', F')
        # Total tokens N = t_p * f_p
        # Expand across combined grid: (t_p, f_p, t_p, f_p, H) -> (t_p * f_p, t_p * f_p, H)
        b_total = b_t[:, None, :, None, :] + b_f[None, :, None, :, :]  # (t, f, t, f, H)
        b_total = b_total.permute(4, 0, 1, 2, 3).reshape(self.num_heads, t_p * f_p, t_p * f_p)
        return b_total.unsqueeze(0)  # (1, H, N, N)


class CEDBlock(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, rel_pos_bias: Tensor | None = None) -> Tensor:
        # Pre-LN Self-Attention
        b, n, c = x.shape
        norm_x = self.norm1(x)
        qkv = self.qkv(norm_x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
            # Handle CLS token: pad bias if n == bias.shape[-1] + 1
            if n == rel_pos_bias.shape[-1] + 1:
                bias = F.pad(rel_pos_bias, (1, 0, 1, 0))
            else:
                bias = rel_pos_bias
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj_drop(self.proj(out))

        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


class CEDTiny(BaseAudioClassifier):
    """
    CED-Tiny Model: 12-layer Vision Transformer (D=192, H=3, 2D Relative Bias).
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 64,
        num_frames: int = 398,
        hidden_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.patch_embed = PatchEmbed(in_chans=1, embed_dim=hidden_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.rel_pos_bias = RelativePositionBias2D(num_heads=num_heads)

        self.blocks = nn.ModuleList([
            CEDBlock(embed_dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: Tensor) -> Tensor:
        # x: (B, 1, F, T)
        tokens, f_p, t_p = self.patch_embed(x)
        b = tokens.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        bias = self.rel_pos_bias(f_p, t_p, x.device)
        for block in self.blocks:
            tokens = block(tokens, bias)

        return self.norm(tokens)

    def forward_head(self, feat: Tensor) -> Tensor:
        # Use CLS token for classification
        cls_feat = feat[:, 0]
        return self.head(cls_feat)

