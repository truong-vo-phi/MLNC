from __future__ import annotations

from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

from .config import BaselineConfig


class CEDViT(nn.Module):
    """ViT trên log-mel với chiều d, số lớp và số head lấy từ cấu hình (CED-Tiny 192/12/3, CED-Mini 256/12/4)."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.network = VisionTransformer(
            img_size=(config.freq_bins, config.num_frames),
            patch_size=16,
            in_chans=1,
            num_classes=config.num_classes,
            embed_dim=config.hidden,
            depth=config.layers,
            num_heads=config.heads,
            mlp_ratio=config.ffn_ratio,
            drop_rate=config.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
