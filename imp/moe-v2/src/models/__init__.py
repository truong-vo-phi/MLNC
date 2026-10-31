"""
Model architectures for Single-Backbone MoE v2:
- Comparative ViT baselines: DeiTTinyBaseline, ViTSSmallBaseline, CEDTinyBaseline
- Single-backbone MoE variants: CEDSingleMoESplitFFN, CEDHydraLoRAMoE
- BEATs Teacher 12L: BEATClassifier, load_fold_teacher
"""

from src.models.beat import BEATClassifier, BEATConfig, load_fold_teacher
from src.models.deit_baseline import CEDTinyBaseline, DeiTTinyBaseline, ViTSSmallBaseline
from src.models.moe_hydra import CEDHydraLoRAMoE, HydraLinear
from src.models.moe_split import CEDSingleMoESplitFFN, SplitMoEFFN
from src.models.vit_bias import BiasAttention, ViTSBias, load_ced_tiny

__all__ = [
    "DeiTTinyBaseline",
    "ViTSSmallBaseline",
    "CEDTinyBaseline",
    "CEDSingleMoESplitFFN",
    "CEDHydraLoRAMoE",
    "SplitMoEFFN",
    "HydraLinear",
    "ViTSBias",
    "BiasAttention",
    "load_ced_tiny",
    "BEATClassifier",
    "BEATConfig",
    "load_fold_teacher",
]
