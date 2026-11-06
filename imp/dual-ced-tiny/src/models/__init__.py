"""
Model Registry for Dual-CED-Tiny Project.
Exports Dual-CED-Tiny, CED-Tiny Low-Expert, and CED-Tiny High-Expert.
"""

from __future__ import annotations

from typing import Any, Type

from src.models.base import BaseAudioClassifier
from src.models.ced_tiny import CEDTiny, load_ced_pretrained
from src.models.dual_ced_tiny import DualCEDTiny

MODEL_REGISTRY: dict[str, Type[BaseAudioClassifier]] = {
    "dual_ced_tiny": DualCEDTiny,
    "ced_tiny_low": CEDTiny,
    "ced_tiny_high": CEDTiny,
}


def list_available_models() -> list[str]:
    """Returns canonical registered model names."""
    return list(MODEL_REGISTRY.keys())


def build_model(
    model_name: str = "dual_ced_tiny",
    num_classes: int = 36,
    sample_rate: int = 16000,
    n_mels: int = 64,
    num_frames: int = 398,
    snr_threshold_db: float = 18.0,
    router_mode: str = "energy_threshold",
    router_transition_center_db: float = 15.0,
    router_transition_tau: float = 2.5,
    router_token_proj_dim: int = 16,
    router_hidden_dim: int = 32,
    use_grep: bool = True,
    post_ln: bool = True,
    pre_ln: bool = True,
    **kwargs: Any,
) -> BaseAudioClassifier:
    """
    Builds model instance by name.
    """
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "dual": "dual_ced_tiny",
        "dual_ced": "dual_ced_tiny",
        "dual_student": "dual_ced_tiny",
        "low": "ced_tiny_low",
        "low_expert": "ced_tiny_low",
        "high": "ced_tiny_high",
        "high_expert": "ced_tiny_high",
    }
    canonical_key = aliases.get(key, key)

    if canonical_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_available_models()}")

    cls = MODEL_REGISTRY[canonical_key]
    if canonical_key == "dual_ced_tiny":
        return cls(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
            snr_threshold_db=snr_threshold_db,
            router_mode=router_mode,
            router_transition_center_db=router_transition_center_db,
            router_transition_tau=router_transition_tau,
            router_token_proj_dim=router_token_proj_dim,
            router_hidden_dim=router_hidden_dim,
            use_grep=use_grep,
            post_ln=post_ln,
            pre_ln=pre_ln,
            **kwargs,
        )

    return cls(
        num_classes=num_classes,
        sample_rate=sample_rate,
        n_mels=n_mels,
        num_frames=num_frames,
        use_grep=use_grep,
        post_ln=post_ln,
        pre_ln=pre_ln,
        **kwargs,
    )


__all__ = [
    "BaseAudioClassifier",
    "CEDTiny",
    "DualCEDTiny",
    "load_ced_pretrained",
    "MODEL_REGISTRY",
    "list_available_models",
    "build_model",
]
