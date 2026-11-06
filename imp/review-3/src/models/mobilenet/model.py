"""
MobileNet Architecture for Audio Classification.
Reference: Howard et al., 'Searching for MobileNetV3', ICCV 2019.
"""

from __future__ import annotations

import timm
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseAudioClassifier


class MobileNet(BaseAudioClassifier):
    """
    Standard MobileNetV3-Large on 1-channel audio spectrograms.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        arch: str = "mobilenetv3_large_100",
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
            arch,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        return x

    def forward_head(self, feat: Tensor) -> Tensor:
        return self.backbone(feat)

