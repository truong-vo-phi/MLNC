"""
Models package for single-backbone CED MoE.
"""

from __future__ import annotations

from src.models.split_moe import CEDSingleMoESplitFFN, SplitMoEFFN
from src.models.vit_grep import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
try:
    from src.models.deit_baseline import CEDTinyBaseline, DeiTTinyBaseline, ViTSSmallBaseline
    from src.models.moe_hydra import CEDHydraLoRAMoE, HydraLinear
except Exception:
    pass

try:
    from src.models.beat import BEATClassifier, BEATConfig, load_fold_teacher
except Exception:
    pass

__all__ = [
    "CEDSingleMoESplitFFN",
    "SplitMoEFFN",
    "ViTSBias",
    "BiasAttention",
    "SharedBias",
    "load_ced_tiny",
    "CEDTinyBaseline",
    "DeiTTinyBaseline",
    "ViTSSmallBaseline",
    "CEDHydraLoRAMoE",
    "BEATClassifier",
]
