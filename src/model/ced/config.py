from __future__ import annotations

from dataclasses import dataclass

from src.core import BaseConfig


@dataclass(frozen=True)
class CEDConfig(BaseConfig):
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 12
    num_heads: int = 4
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    target_length: int = 1012
    n_mels: int = 64
    num_classes: int = 36
