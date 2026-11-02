"""
PANN-CNN 10 Architecture for Audio Classification.
Reference: Kong et al., 'PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition', IEEE/ACM TASLP 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseAudioClassifier


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class PANNCNN10(BaseAudioClassifier):
    """
    PANNs CNN-10: 4-stage convolutional backbone with dual pooling head.
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 128,
        num_frames: int = 398,
        channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels,
            num_frames=num_frames,
        )
        self.channels = tuple(channels)
        blocks = []
        in_c = 1
        for out_c in self.channels:
            blocks.append(ConvBlock(in_c, out_c))
            in_c = out_c
        self.backbone = nn.Sequential(*blocks)

        final_c = self.channels[-1]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_c, final_c)
        self.head = nn.Linear(final_c, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        # x: (B, 1, F, T) -> (B, 512, F', T')
        return self.backbone(x)

    def forward_head(self, feat: Tensor) -> Tensor:
        # Dual pooling: mean across freq, (max + mean) across time
        x = feat.mean(dim=2)  # (B, C, T')
        pooled = x.max(dim=2).values + x.mean(dim=2)  # (B, C)
        x = self.dropout(pooled)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.head(x)
        return logits

