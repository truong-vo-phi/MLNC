"""
CED-Small: Consistent Ensemble Distillation Small Audio Transformer.
Reference: Dinkel et al., 'CED: Consistent Ensemble Distillation for Audio Tagging', ICASSP 2024.
Config: D=384, 12 layers, 6 heads, MLP ratio 4.0, ~21.4M params.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseAudioClassifier
from src.models.ced_tiny.model import CEDBlock, PatchEmbed, RelativePositionBias2D


class CEDSmall(BaseAudioClassifier):
    """
    CED-Small Model: 12-layer Audio Transformer with D=384, H=6, 2D Relative Position Bias.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 64,
        num_frames: int = 398,
        hidden_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
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
        tokens, f_p, t_p = self.patch_embed(x)
        b = tokens.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        bias = self.rel_pos_bias(f_p, t_p, x.device)
        for block in self.blocks:
            tokens = block(tokens, bias)

        return self.norm(tokens)

    def forward_head(self, feat: Tensor) -> Tensor:
        cls_feat = feat[:, 0]
        return self.head(cls_feat)

