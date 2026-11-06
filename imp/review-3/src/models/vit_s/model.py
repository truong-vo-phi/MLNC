"""
ViT-S (Vision Transformer Small) for Audio Spectrograms.
Reference: Touvron et al., 'Training data-efficient image transformers & distillation through attention' (DeiT/ViT-S), ICML 2021.
Config: 12 layers, D=384, 6 heads, MLP ratio 4.0, ~21.4M params.
"""

from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseAudioClassifier


class ViTS(BaseAudioClassifier):
    """
    ViT-S (Vision Transformer Small / DeiT-Small) adapted for 1-channel audio spectrograms.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            in_chans=1,
            img_size=(n_mels, num_frames),
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return x

    def forward_head(self, feat: Tensor) -> Tensor:
        return self.backbone(feat)

