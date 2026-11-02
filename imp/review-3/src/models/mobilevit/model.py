"""
MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer for Audio.
Reference: Mehta & Rastegari, 'MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer', ICLR 2022.
"""

from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseAudioClassifier


class MobileViT(BaseAudioClassifier):
    """
    MobileViT-S Architecture combining CNN local representations with Transformer global attention.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.backbone = timm.create_model(
            "mobilevit_s",
            pretrained=False,
            in_chans=1,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return x

    def forward_head(self, feat: Tensor) -> Tensor:
        return self.backbone(feat)

