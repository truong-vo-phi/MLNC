"""
Efficient-mn10 Architecture for Audio Pattern Recognition.
Reference: Schmid et al., 'Efficient Large-Scale Audio Tagging via Transformer-to-CNN Knowledge Distillation' (EfficientAT), Interspeech 2023.
Architecture: MobileNetV3-Large (width_mult=1.0) on 1-channel audio spectrograms.
"""

from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseAudioClassifier


class EfficientMN10(BaseAudioClassifier):
    """
    Efficient-mn10 (MobileNetV3-Large 1.0) for Audio Spectrograms.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return x

    def forward_head(self, feat: Tensor) -> Tensor:
        return self.backbone(feat)

