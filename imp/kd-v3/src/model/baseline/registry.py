from __future__ import annotations

from torch import nn

from .cnn10 import CNN10
from .config import BaselineConfig
from .conformer import ConformerNet
from .mobilenet import MobileNetV3Width
from .recurrent import CRNN, BiLSTM
from .timm_backbone import TimmBackbone
from .vit import CEDViT

CUSTOM_BACKBONES: dict[str, type[nn.Module]] = {
    "cnn10": CNN10,
    "crnn": CRNN,
    "bilstm": BiLSTM,
    "conformer": ConformerNet,
    "mobilenetv3_width": MobileNetV3Width,
    "vit_ced": CEDViT,
}


# Trọng số được nạp ngoài build_backbone (CNN10.load_panns), nên pretrained: true vẫn hợp lệ.
PRETRAINABLE = ("cnn10",)


def build_backbone(config: BaselineConfig) -> nn.Module:
    if config.arch in CUSTOM_BACKBONES:
        if config.pretrained and config.arch not in PRETRAINABLE:
            raise ValueError(f"'{config.arch}' has no pretrained weights, set pretrained: false")
        return CUSTOM_BACKBONES[config.arch](config)
    return TimmBackbone(config)
