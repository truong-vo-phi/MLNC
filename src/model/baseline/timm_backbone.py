from __future__ import annotations

import timm
from torch import Tensor, nn

from .config import BaselineConfig

PATCH_PREFIXES = ("deit", "vit", "swin")
WINDOW_PREFIXES = ("swin",)
STAGE_PREFIXES = ("deit", "vit")
STAGE_COUNT = 4
PATCH_SIZE = 16


class TimmBackbone(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        kwargs: dict[str, object] = {}
        self.frames = config.num_frames
        if config.arch.startswith(STAGE_PREFIXES):
            self.frames = config.num_frames // PATCH_SIZE * PATCH_SIZE  # bỏ phần dư của khung thời gian
        if config.arch.startswith(PATCH_PREFIXES):
            kwargs["img_size"] = (config.freq_bins, self.frames)
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
        return self.network(x[..., : self.frames])

    def forward_stages(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Logit và đặc trưng (B, C, H, W) ở 4 khối ViT cách đều nhau, khối cuối là khối thứ `depth`."""
        if not hasattr(self.network, "forward_intermediates"):
            raise ValueError(f"{type(self.network).__name__} does not expose intermediate features")
        depth = len(self.network.blocks)
        indices = [depth * (stage + 1) // STAGE_COUNT - 1 for stage in range(STAGE_COUNT)]
        final, intermediates = self.network.forward_intermediates(x[..., : self.frames], indices=indices, norm=False)
        return self.network.forward_head(final), tuple(intermediates)
