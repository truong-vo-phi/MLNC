from __future__ import annotations

import torch
from torch import nn

from .config import CEDConfig
from .encoder import Block
from .patch_embed import PatchEmbed


class CEDBackbone(nn.Module):
    """log-mel (B, 64, T) -> patch features (B, N, 256), N = (64/16) * (T/16)."""

    def __init__(self, config: CEDConfig):
        super().__init__()
        self.init_bn = nn.BatchNorm2d(config.n_mels, momentum=0.01)
        self.patch_embed = PatchEmbed(config.patch_size, config.embed_dim)
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, config.embed_dim, 1, config.target_length // config.patch_size) * 0.02
        )
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, config.embed_dim, config.n_mels // config.patch_size, 1) * 0.02
        )
        self.pos_drop = nn.Dropout(config.drop_rate)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        x = self.init_bn(spectrogram.unsqueeze(2)).transpose(1, 2)  # (B, 1, F, T)
        x = self.patch_embed(x)  # (B, C, F/p, T/p)
        frames, limit = x.shape[-1], self.time_pos_embed.shape[-1]
        if frames > limit:
            raise ValueError(f"input has {frames} time patches, the model supports at most {limit}")
        x = x + self.time_pos_embed[..., :frames] + self.freq_pos_embed
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        return self.norm(self.blocks(x))
