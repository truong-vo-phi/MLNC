"""
CED-Tiny with BEATs Gated Relative Attention Bias and DeepNorm Post-LN.
Variant specified for Task 3 comparison:
- Gated Relative Attention Bias (BEATs GREP GRU-style gate scale)
- DeepNorm Post-LN (alpha = (2*L)^0.25 ≈ 2.2133, Post-LayerNorm residuals)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseAudioClassifier
from src.models.ced_tiny.model import PatchEmbed


def relative_position_bucket(relative_position: Tensor, num_buckets: int = 320, max_distance: int = 800) -> Tensor:
    """Computes relative position buckets identical to BEATs."""
    ret = 0
    n = -relative_position
    n = torch.max(n, torch.zeros_like(n))
    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        torch.log(n.float() / max_exact + 1e-6)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


class SharedBiasTable:
    def __init__(self) -> None:
        self.value: Tensor | None = None


class GatedBEATsBiasAttention(nn.Module):
    """
    Self-Attention module equipped with BEATs Gated Relative Attention Bias:
    Scale = gate_a * (gate_b * grep_a - 1.0) + 2.0
    attn_bias = Scale * PositionBias
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        num_buckets: int = 320,
        max_dist: int = 800,
        gate_hidden: int = 8,
        shared_table: SharedBiasTable | None = None,
        owns_table: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.gate_hidden = gate_hidden
        self.num_buckets = num_buckets
        self.max_dist = max_dist

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.shared = shared_table if shared_table is not None else SharedBiasTable()
        if owns_table:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
            nn.init.zeros_(self.relative_attention_bias.weight)

        # GRU-style gate for position bias scaling
        self.grep_linear = nn.Linear(self.head_dim, gate_hidden)
        self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def compute_bias(self, tokens: int, device: torch.device) -> Tensor:
        # tokens includes CLS token
        pos = torch.arange(tokens - 1, device=device)
        rel_diff = pos[None, :] - pos[:, None]
        bucket = relative_position_bucket(rel_diff, self.num_buckets, self.max_dist)
        bias = self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)  # (1, H, N-1, N-1)
        # Pad for CLS token (1 at start)
        return F.pad(bias, (1, 0, 1, 0))

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Shared relative bias calculation
        if hasattr(self, "relative_attention_bias"):
            self.shared.value = self.compute_bias(n, x.device)
        pos_bias = self.shared.value

        # Compute dynamic gate scaling: (B, H, N, 1)
        gate = torch.sigmoid(self.grep_linear(q).view(b, self.num_heads, n, 2, self.gate_hidden // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)
        scale_mod = gate_a * (gate_b * self.grep_a - 1.0) + 2.0

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if pos_bias is not None:
            effective_bias = (scale_mod * pos_bias).to(q.dtype)
            attn = attn + effective_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class DeepNormPostLNBlock(nn.Module):
    """
    DeepNorm Post-LayerNorm Transformer Block:
    x = PostLN1(alpha * x + Sublayer1(x))
    x = PostLN2(alpha * x + Sublayer2(x))
    where alpha = (2 * total_layers)^0.25
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        alpha: float = 2.2133,
        shared_table: SharedBiasTable | None = None,
        owns_table: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.attn = GatedBEATsBiasAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            shared_table=shared_table,
            owns_table=owns_table,
            dropout=dropout,
        )
        self.post_norm1 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.post_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Post-LN with DeepNorm residual scale
        attn_out = self.attn(x)
        x = self.post_norm1(self.alpha * x + attn_out)

        mlp_out = self.mlp(x)
        x = self.post_norm2(self.alpha * x + mlp_out)
        return x


class CEDTinyBeatBiasPostLN(BaseAudioClassifier):
    """
    CED-Tiny with BEATs Gated Relative Attention Bias + DeepNorm Post-LN.
    12 layers, D=192, H=3, alpha ≈ 2.2133, ~5.40M params.
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
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        alpha = math.pow(2.0 * num_layers, 0.25)  # ≈ 2.21336
        shared_table = SharedBiasTable()

        self.blocks = nn.ModuleList([
            DeepNormPostLNBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                alpha=alpha,
                shared_table=shared_table,
                owns_table=(i == 0),
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        tokens, _, _ = self.patch_embed(x)
        b = tokens.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)

    def forward_head(self, feat: Tensor) -> Tensor:
        cls_feat = feat[:, 0]
        return self.head(cls_feat)

