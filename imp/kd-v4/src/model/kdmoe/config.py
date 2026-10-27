from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig

HYBRID_BRANCHES = ("both", "attn", "gru")
FREQ_POOL_MODES = ("mean", "attn")
INFERENCE_MODES = ("weighted", "top1")


@dataclass(frozen=True)
class KDMoEConfig(BaseConfig):
    section: ClassVar[str] = "model"

    d: int = 128
    heads: int = 4
    n_hybrid: int = 2
    n_vit: int = 2
    ffn_ratio: int = 4
    hybrid_gru_hidden: int = 64
    final_gru_hidden: int = 64
    expert_hidden: int = 128
    router_hidden: int = 64
    gate_eps: float = 0.05
    gate_bias_init: float = 2.0
    dropout: float = 0.1
    drop_path_max: float = 0.05
    patch_size: int = 16
    freq_bins: int = 128
    num_frames: int = 384
    num_classes: int = 36
    num_groups: int = 4
    use_gate: bool = True
    hybrid_branches: str = "both"
    freq_pool: str = "mean"
    inference_mode: str = "weighted"
    specaug_freq_masks: int = 2
    specaug_freq_width: int = 16
    specaug_time_masks: int = 2
    specaug_time_width: int = 24

    def __post_init__(self) -> None:
        if self.hybrid_branches not in HYBRID_BRANCHES:
            raise ValueError(f"hybrid_branches must be one of {HYBRID_BRANCHES}, got {self.hybrid_branches!r}")
        if self.freq_pool not in FREQ_POOL_MODES:
            raise ValueError(f"freq_pool must be one of {FREQ_POOL_MODES}, got {self.freq_pool!r}")
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"inference_mode must be one of {INFERENCE_MODES}, got {self.inference_mode!r}")
        if self.freq_bins % self.patch_size or self.num_frames % self.patch_size:
            raise ValueError(
                f"freq_bins and num_frames must be divisible by patch_size {self.patch_size}, "
                f"got {self.freq_bins} and {self.num_frames}"
            )
