from __future__ import annotations

import timm
from torch import Tensor, nn

from .config import BaselineConfig

PATCH_PREFIXES = ("deit", "vit", "swin")
WINDOW_PREFIXES = ("swin",)


class TimmBackbone(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        kwargs: dict[str, object] = {}
        if config.arch.startswith(PATCH_PREFIXES):
            kwargs["img_size"] = (config.freq_bins, config.num_frames)
        if config.arch.startswith(WINDOW_PREFIXES):
            kwargs["window_size"] = config.window_size
        self.network = timm.create_model(
            config.arch,
            pretrained=config.pretrained,
            in_chans=1,
            num_classes=config.num_classes,
            drop_rate=config.dropout,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
