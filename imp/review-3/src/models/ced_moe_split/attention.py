"""
2D Gated Relative Position Bias Attention (GREP Attention) for CED-Tiny Backbone.

Features:
1. Shared 2D Relative Position Bias Table P_rel (calculated once per forward pass by Block 0, reused by Blocks 1-11).
2. Gated Relative Position (GREP):
   - Query Q goes through Linear(64 -> 8) + Sigmoid -> g_a, g_b
   - Scaling factor: S = g_a * (g_b * grep_a - 1.0) + 2.0
   - Modulated bias: B_attn = S * P_rel added directly to QK^T / sqrt(64) before softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GATE_HIDDEN = 8


class SharedBias:
    """Shared container for relative position bias matrix computed once per forward pass."""
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """
    2D Gated Relative Position Bias Attention (GREP Attention).
    3 attention heads, head_dim = 64 (total embed_dim = 192).
    """
    def __init__(
        self,
        base: nn.Module,
        shared: SharedBias,
        owns_table: bool,
        freq_patches: int = 4,
        time_patches: int = 24,
        has_cls: bool = True,
        bias_init: str = "zero",
    ) -> None:
        super().__init__()
        self.has_cls = has_cls
        self.num_heads = base.num_heads
        self.head_dim = base.head_dim
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.shared = shared
        self.freq_patches = freq_patches
        self.time_patches = time_patches
        n_patches = freq_patches * time_patches

        if owns_table:
            num_f = 2 * freq_patches - 1
            num_t = 2 * time_patches - 1
            self.freq_rel_bias = nn.Embedding(num_f, self.num_heads)
            self.time_rel_bias = nn.Embedding(num_t, self.num_heads)
            self.cls_to_patches = nn.Parameter(torch.zeros(1, self.num_heads, 1, n_patches))
            self.patches_to_cls = nn.Parameter(torch.zeros(1, self.num_heads, n_patches, 1))
            self.cls_to_cls = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

            if bias_init in {"normal", "random"}:
                nn.init.normal_(self.freq_rel_bias.weight, std=0.01)
                nn.init.normal_(self.time_rel_bias.weight, std=0.01)
                nn.init.normal_(self.cls_to_patches, std=0.01)
                nn.init.normal_(self.patches_to_cls, std=0.01)
                nn.init.normal_(self.cls_to_cls, std=0.01)
            else:
                self.freq_rel_bias.weight.data.zero_()
                self.time_rel_bias.weight.data.zero_()

        # GREP linear gating (64 -> 8)
        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def compute_bias(self, device: torch.device) -> Tensor:
        f_p, t_p = self.freq_patches, self.time_patches
        grid_f = torch.arange(f_p, device=device).unsqueeze(1).expand(f_p, t_p).reshape(-1)
        grid_t = torch.arange(t_p, device=device).unsqueeze(0).expand(f_p, t_p).reshape(-1)
        delta_f = (grid_f.unsqueeze(1) - grid_f.unsqueeze(0)) + (f_p - 1)
        delta_t = (grid_t.unsqueeze(1) - grid_t.unsqueeze(0)) + (t_p - 1)

        bias_patch = (self.freq_rel_bias(delta_f) + self.time_rel_bias(delta_t)).permute(2, 0, 1).unsqueeze(0)
        if not self.has_cls:
            return bias_patch

        row_cls = torch.cat([self.cls_to_cls, self.cls_to_patches], dim=-1)
        rows_patch = torch.cat([self.patches_to_cls, bias_patch], dim=-1)
        return torch.cat([row_cls, rows_patch], dim=-2)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)

        if hasattr(self, "freq_rel_bias"):
            self.shared.value = self.compute_bias(x.device)
        position_bias = self.shared.value

        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        bias = (scale * position_bias).to(q.dtype) if position_bias is not None else None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))

