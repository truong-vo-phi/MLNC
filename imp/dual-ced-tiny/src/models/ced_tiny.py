"""
Single CED-Tiny Audio Transformer Backbone (~5.40M parameters).
Reference: Dinkel et al., 'CED: Consistent Ensemble Distillation for Audio Tagging', ICASSP 2024.
Matches official `mispeech/ced-tiny` state_dict keys (148 pretrained backbone tensors).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseAudioClassifier

GREP_HIDDEN = 8


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        # x: (B, 1, F, T) -> (B, embed_dim, F', T')
        x = self.proj(x)
        f_p, t_p = x.shape[2], x.shape[3]
        # Time-major ordering: (B, embed_dim, F', T') -> (B, T'*F', embed_dim)
        x = x.permute(0, 3, 2, 1).flatten(1, 2)
        return x, f_p, t_p


class RelativePositionBias2D(nn.Module):
    def __init__(
        self,
        num_heads: int,
        freq_patches: int,
        time_patches: int,
        max_dist_f: int = 16,
        max_dist_t: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_dist_f = max_dist_f
        self.max_dist_t = max_dist_t
        self.bias_f = nn.Embedding(2 * max_dist_f + 1, num_heads)
        self.bias_t = nn.Embedding(2 * max_dist_t + 1, num_heads)
        num_patches = freq_patches * time_patches
        self.cls_to_patches = nn.Parameter(torch.zeros(1, num_heads, 1, num_patches))
        self.patches_to_cls = nn.Parameter(torch.zeros(1, num_heads, num_patches, 1))
        self.cls_to_cls = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        nn.init.zeros_(self.bias_f.weight)
        nn.init.zeros_(self.bias_t.weight)

    def forward(self, f_p: int, t_p: int, device: torch.device) -> Tensor:
        coords_f = torch.arange(f_p, device=device)
        coords_t = torch.arange(t_p, device=device)
        diff_f = (coords_f[:, None] - coords_f[None, :] + self.max_dist_f).clamp(0, 2 * self.max_dist_f)
        diff_t = (coords_t[:, None] - coords_t[None, :] + self.max_dist_t).clamp(0, 2 * self.max_dist_t)

        b_f = self.bias_f(diff_f)  # (f, f, H)
        b_t = self.bias_t(diff_t)  # (t, t, H)

        # Composite relative bias across 2D grid: (t_p * f_p, t_p * f_p, H)
        b_total = b_t[:, None, :, None, :] + b_f[None, :, None, :, :]
        b_total = b_total.permute(4, 0, 1, 2, 3).reshape(self.num_heads, t_p * f_p, t_p * f_p)
        row_cls = torch.cat([self.cls_to_cls, self.cls_to_patches[..., : t_p * f_p]], dim=-1)
        rows_patch = torch.cat([self.patches_to_cls[..., : t_p * f_p, :], b_total.unsqueeze(0)], dim=-1)
        return torch.cat([row_cls, rows_patch], dim=-2)  # (1, H, N+1, N+1)


class Attention(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 3, dropout: float = 0.0, use_grep: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.use_grep = use_grep
        if use_grep:
            self.grep_linear = nn.Linear(self.head_dim, GREP_HIDDEN)
            self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def forward(self, x: Tensor, rel_pos_bias: Tensor | None = None) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
            bias = rel_pos_bias
            if self.use_grep:
                gate = torch.sigmoid(self.grep_linear(q).view(b, self.num_heads, n, 2, GREP_HIDDEN // 2).sum(-1))
                gate_a, gate_b = gate.chunk(2, dim=-1)
                scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                bias = scale * bias
            attn = attn + bias

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class CEDBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_grep: bool = True,
        post_ln: bool = True,
        pre_ln: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim) if pre_ln else nn.Identity()
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, use_grep=use_grep)
        self.norm2 = nn.LayerNorm(embed_dim) if pre_ln else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), drop=dropout)
        self.post_ln = post_ln
        self.alpha = alpha
        if post_ln:
            self.post_norm1 = nn.LayerNorm(embed_dim)
            self.post_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, rel_pos_bias: Tensor | None = None) -> Tensor:
        if self.post_ln:
            # Alpha-scaled residuals and post-residual LayerNorm, following the
            # AlphaBlock/DeepNorm implementation used by imp/moe and imp/moe-v2.
            x = self.post_norm1(self.alpha * x + self.attn(self.norm1(x), rel_pos_bias))
            x = self.post_norm2(self.alpha * x + self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x), rel_pos_bias)
            x = x + self.mlp(self.norm2(x))
        return x


class CEDTiny(BaseAudioClassifier):
    """
    CED-Tiny Audio Transformer:
      - 12 Transformer layers (D=192, H=3, MLP=768)
      - Query-gated 2D relative position bias (GREP)
      - Alpha-scaled residuals with optional Post-LN
      - Patch size 16x16
      - Linear Classifier Head (36 classes)
      - With GREP + Post-LN: approximately 5.412M parameters per expert
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
        use_grep: bool = True,
        post_ln: bool = True,
        pre_ln: bool = True,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.patch_embed = PatchEmbed(in_chans=1, embed_dim=hidden_dim, patch_size=patch_size)
        self.use_grep = use_grep
        self.post_ln = post_ln
        self.pre_ln = pre_ln
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.rel_pos_bias = RelativePositionBias2D(
            num_heads=num_heads,
            freq_patches=n_mels // patch_size,
            time_patches=num_frames // patch_size,
        )

        alpha = (2 * num_layers) ** 0.25
        self.blocks = nn.ModuleList([
            CEDBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_grep=use_grep,
                post_ln=post_ln,
                pre_ln=pre_ln,
                alpha=alpha,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: Tensor) -> Tensor:
        tokens, f_p, t_p = self.patch_embed(x)
        b = tokens.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        bias = self.rel_pos_bias(f_p, t_p, x.device)
        for block in self.blocks:
            tokens = block(tokens, bias)

        return self.norm(tokens)

    def forward_with_context(self, x: Tensor, context_block: int = 9) -> tuple[Tensor, Tensor]:
        """Return class logits and the CLS token after ``context_block``.

        The intermediate token feeds the self-contained spectral-token router.
        Both outputs come from one traversal of this expert's transformer.
        """
        if not 0 <= context_block < len(self.blocks):
            raise ValueError(f"context_block must be in [0, {len(self.blocks) - 1}]")
        tokens, f_p, t_p = self.patch_embed(x)
        tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        bias = self.rel_pos_bias(f_p, t_p, x.device)
        context: Tensor | None = None
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, bias)
            if index == context_block:
                context = tokens[:, 0]
        if context is None:
            raise RuntimeError("failed to capture the requested intermediate CED token")
        logits = self.head(self.norm(tokens)[:, 0])
        return logits, context

    def forward_head(self, feat: Tensor) -> Tensor:
        # Use CLS token representation
        cls_feat = feat[:, 0]
        return self.head(cls_feat)


def load_ced_pretrained(model: CEDTiny, repo_or_path: str | Path = "mispeech/ced-tiny") -> int:
    """
    Loads official AudioSet-pretrained `mispeech/ced-tiny` backbone weights into a `CEDTiny` instance.
    Returns the number of matched tensors loaded (148 for standard CED-Tiny).
    """
    raw_state: dict[str, Any] = {}
    cand = Path(repo_or_path)
    if cand.is_file():
        if cand.suffix == ".safetensors":
            from safetensors.torch import load_file
            raw_state = load_file(str(cand))
        else:
            raw_state = torch.load(cand, map_location="cpu", weights_only=False)
    else:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sf_path = hf_hub_download(repo_id=str(repo_or_path), filename="model.safetensors")
        raw_state = load_file(sf_path)

    if "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    elif "model" in raw_state and isinstance(raw_state["model"], dict):
        raw_state = raw_state["model"]

    target_state = model.state_dict()
    matched: dict[str, Tensor] = {}
    for k, v in raw_state.items():
        clean_k = k.replace("encoder.", "").replace("backbone.", "")
        if clean_k in target_state and target_state[clean_k].shape == v.shape:
            matched[clean_k] = v

    model.load_state_dict(matched, strict=False)
    return len(matched)
