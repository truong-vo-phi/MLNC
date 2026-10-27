from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import mobilenet_v3_large

from .config import BaselineConfig


class MobileNetV3Width(nn.Module):
    """MobileNetV3-Large với hệ số độ rộng `width_mult` (0,4 / 0,5 / 1,0 / 2,0 ứng với mn04 / mn05 / mn10 / mn20)."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.network = mobilenet_v3_large(
            weights=None, width_mult=config.width_mult, num_classes=config.num_classes, dropout=config.dropout
        )
        stem = self.network.features[0][0]
        self.network.features[0][0] = nn.Conv2d(
            1, stem.out_channels, kernel_size=stem.kernel_size, stride=stem.stride, padding=stem.padding, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
