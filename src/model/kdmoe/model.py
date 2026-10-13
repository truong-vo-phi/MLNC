from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml
from torch import Tensor, nn

from src.core import BaseModel

from .config import KDMoEConfig
from .experts import Experts
from .frontend import LogMel, SpecAugment
from .gate import MaskGate, NoiseHead
from .hybrid_block import HybridBlock
from .patch import PatchEmbed
from .pool_gru import FreqPoolGRU
from .router import Router
from .vit import ViTBlock


class KDMoEOutput(NamedTuple):
    logits: Tensor
    router_logits: Tensor
    e: Tensor
    gate: Tensor
    noise_mel: Tensor | None = None
    noise_target: Tensor | None = None


class KDMoEModel(BaseModel):
    """Student lai cascade: khối lai x2 -> cổng mặt nạ -> ViT -> gộp tần số -> BiGRU -> router + 4 expert."""

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config
        frontend = {"num_frames": config.num_frames, **(frontend_config or {})}
        self.frontend = LogMel(**frontend)
        self.spec_augment = SpecAugment(
            freq_masks=config.specaug_freq_masks,
            freq_width=config.specaug_freq_width,
            time_masks=config.specaug_time_masks,
            time_width=config.specaug_time_width,
        )
        self.patch_embed = PatchEmbed(config.d, config.patch_size, config.freq_bins, config.num_frames)
        total_blocks = config.n_hybrid + config.n_vit

        def drop_path(index: int) -> float:
            return config.drop_path_max * index / max(total_blocks - 1, 1)

        self.hybrid = nn.ModuleList(
            HybridBlock(
                config.d,
                config.heads,
                config.hybrid_gru_hidden,
                config.ffn_ratio,
                config.dropout,
                drop_path(index),
                config.hybrid_branches,
            )
            for index in range(config.n_hybrid)
        )
        self.gate = MaskGate(config.d, config.gate_eps, config.gate_bias_init) if config.use_gate else None
        self.noise_head = NoiseHead(config.d, config.patch_size, config.freq_bins, config.num_frames)
        self.vit = nn.ModuleList(
            ViTBlock(config.d, config.heads, config.ffn_ratio, config.dropout, drop_path(config.n_hybrid + index))
            for index in range(config.n_vit)
        )
        self.final_ln = nn.LayerNorm(config.d)
        self.pool_gru = FreqPoolGRU(config.d, config.final_gru_hidden, freq_pool=config.freq_pool)
        embed_dim = 4 * config.final_gru_hidden
        self.router = Router(embed_dim, config.router_hidden, config.num_groups)
        self.experts = Experts(
            embed_dim, config.expert_hidden, config.num_classes, config.num_groups, dropout=config.dropout
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> KDMoEModel:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(KDMoEConfig.from_dict(raw[KDMoEConfig.section]), raw.get("frontend"))

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return dict(checkpoint.get("model", checkpoint))

    def forward(self, waveform: Tensor, noise: Tensor | None = None) -> KDMoEOutput:
        x, noise_target = self.frontend(waveform, noise)
        x = self.spec_augment(x)
        h = self.patch_embed(x)
        for block in self.hybrid:
            h = block(h)
        if self.gate is not None:
            h, gate = self.gate(h)
        else:
            gate = torch.ones_like(h)
        noise_mel = self.noise_head(h) if noise_target is not None else None
        for block in self.vit:
            h = block(h)
        h = self.final_ln(h)
        embedding, _ = self.pool_gru(h)
        router_logits = self.router(embedding)
        expert_logits = self.experts(embedding)
        if not self.training and self.config.inference_mode == "top1":
            index = router_logits.argmax(dim=-1)
            logits = expert_logits.gather(1, index[:, None, None].expand(-1, 1, expert_logits.shape[-1])).squeeze(1)
        else:
            weights = router_logits.softmax(dim=-1)
            logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        return KDMoEOutput(
            logits=logits,
            router_logits=router_logits,
            e=embedding,
            gate=gate,
            noise_mel=noise_mel,
            noise_target=noise_target,
        )
