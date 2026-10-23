from __future__ import annotations

from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def conv_stack(channels: tuple[int, ...], dropout: float = 0.0) -> nn.Sequential:
    widths = (1, *channels)
    return nn.Sequential(*(ConvBlock(widths[i], widths[i + 1], dropout) for i in range(len(channels))))
