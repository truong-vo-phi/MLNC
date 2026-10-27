from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def resize_map(x: Tensor, size: tuple[int, int], mode: str) -> Tensor:
    size = (int(size[0]), int(size[1]))
    if tuple(x.shape[-2:]) == size:
        return x
    shrinks = x.shape[-2] >= size[0] and x.shape[-1] >= size[1]
    if mode == "area" and shrinks:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size, mode="nearest")


class ABF(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, fuse: bool, resize: str) -> None:
        super().__init__()
        self.resize = resize
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = nn.Sequential(nn.Conv2d(mid_channels * 2, 2, kernel_size=1), nn.Sigmoid()) if fuse else None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x: Tensor, out_size: tuple[int, int], residual: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.conv1(x)
        if self.attention is not None:
            if residual is None:
                raise ValueError("a fusing ABF needs the residual of the deeper stage")
            residual = F.interpolate(residual, x.shape[-2:], mode="nearest")
            weights = self.attention(torch.cat([x, residual], dim=1))
            x = x * weights[:, :1] + residual * weights[:, 1:]
        x = resize_map(x, out_size, self.resize)
        return self.conv2(x), x


class ReviewModule(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        out_size: tuple[int, int],
        mid_channels: int = 512,
        resize: str = "area",
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("ReviewModule needs at least two student stages")
        self.out_size = tuple(out_size)
        mid = min(mid_channels, in_channels[-1])
        last = len(in_channels) - 1
        blocks = [ABF(channels, mid, out_channels, index < last, resize) for index, channels in enumerate(in_channels)]
        self.abfs = nn.ModuleList(blocks[::-1])

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.abfs):
            raise ValueError(f"expected {len(self.abfs)} stage features, got {len(features)}")
        deepest_first = list(features)[::-1]
        output, residual = self.abfs[0](deepest_first[0], self.out_size)
        outputs = [output]
        for feature, abf in zip(deepest_first[1:], self.abfs[1:], strict=True):
            output, residual = abf(feature, self.out_size, residual)
            outputs.insert(0, output)
        return torch.stack(outputs)
