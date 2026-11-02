"""
CED-SingleMoE-SplitFFN Model Package (~6.62M Parameters).
"""

from src.models.ced_moe_split.acoustic_frontend import (
    AcousticFeatureExtractor,
    LogMel,
    SpecAugment,
    rms_normalize,
)
from src.models.ced_moe_split.attention import BiasAttention, SharedBias
from src.models.ced_moe_split.model import CEDSingleMoESplitFFN
from src.models.ced_moe_split.router import SelfContainedSpectralTokenRouter
from src.models.ced_moe_split.split_ffn import SplitMoEFFN
from src.models.ced_moe_split.trainer import MoETrainer

__all__ = [
    "AcousticFeatureExtractor",
    "BiasAttention",
    "CEDSingleMoESplitFFN",
    "LogMel",
    "MoETrainer",
    "SelfContainedSpectralTokenRouter",
    "SharedBias",
    "SpecAugment",
    "SplitMoEFFN",
    "rms_normalize",
]
