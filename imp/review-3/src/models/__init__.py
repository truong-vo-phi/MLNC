"""
Review-3 Models Registry: Exports all 9 Audio Classification Models.
Each model is organized in its own standalone directory.
"""

from __future__ import annotations

from typing import Any, Type

import yaml
from pathlib import Path

from src.models.base import BaseAudioClassifier
from src.models.beats import BEATs
from src.models.ced_mini import CEDMini
from src.models.ced_moe_split import CEDSingleMoESplitFFN
from src.models.ced_small import CEDSmall
from src.models.ced_tiny import CEDTiny, CEDTinyBeatBiasPostLN
from src.models.efficient_mn10 import EfficientMN10
from src.models.mobilenet import MobileNet
from src.models.mobilevit import MobileViT
from src.models.pann_cnn10 import PANNCNN10
from src.models.vit_s import ViTS

MODEL_REGISTRY: dict[str, Type[BaseAudioClassifier]] = {
    "pann_cnn10": PANNCNN10,
    "ced_tiny": CEDTiny,
    "ced_tiny_beat_bias_postln": CEDTinyBeatBiasPostLN,
    "ced_mini": CEDMini,
    "ced_small": CEDSmall,
    "efficient_mn10": EfficientMN10,
    "vit_s": ViTS,
    "mobilevit": MobileViT,
    "mobilenet": MobileNet,
    "beats": BEATs,
    "ced_single_moe_split": CEDSingleMoESplitFFN,
}


def list_available_models() -> list[str]:
    """Returns list of canonical model keys."""
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str,
    num_classes: int = 36,
    sample_rate: int = 16000,
    n_mels: int = 128,
    num_frames: int = 398,
    **kwargs: Any,
) -> BaseAudioClassifier:
    """
    Builds a model by key name.
    """
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    # Aliases
    aliases = {
        "cnn10": "pann_cnn10",
        "pann_cnn_10": "pann_cnn10",
        "efficient_mn10": "efficient_mn10",
        "efficientat_mn10": "efficient_mn10",
        "vit_small": "vit_s",
        "deit_small": "vit_s",
        "beat": "beats",
        "ced_tiny_postln": "ced_tiny_beat_bias_postln",
        "ced_tiny_bias_postln": "ced_tiny_beat_bias_postln",
        "ced_moe": "ced_single_moe_split",
        "ced_split_moe": "ced_single_moe_split",
        "ced_moe_split": "ced_single_moe_split",
        "moe": "ced_single_moe_split",
        "moe_v2": "ced_single_moe_split",
    }
    key = aliases.get(key, key)

    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_name}' not found. Available models: {list_available_models()}"
        )

    # Some models prefer specific mel bins (e.g. CED uses 64 mels by default)
    if key in ("ced_tiny", "ced_tiny_beat_bias_postln", "ced_mini", "ced_small", "ced_single_moe_split") and "n_mels" not in kwargs:
        n_mels = 64

    model_cls = MODEL_REGISTRY[key]
    return model_cls(
        num_classes=num_classes,
        sample_rate=sample_rate,
        n_mels=n_mels,
        num_frames=num_frames,
        **kwargs,
    )


__all__ = [
    "BaseAudioClassifier",
    "PANNCNN10",
    "CEDTiny",
    "CEDTinyBeatBiasPostLN",
    "CEDMini",
    "CEDSmall",
    "EfficientMN10",
    "ViTS",
    "MobileViT",
    "MobileNet",
    "BEATs",
    "CEDSingleMoESplitFFN",
    "MODEL_REGISTRY",
    "list_available_models",
    "build_model",
]

