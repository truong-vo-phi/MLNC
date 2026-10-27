from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from src.core import LoadReport

from .config import BaselineConfig
from .conv_block import conv_stack
from .panns import convert_panns_cnn10


class CNN10(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        width = config.channels[-1]
        self.blocks = conv_stack(config.channels, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(width, width)
        self.head = nn.Linear(width, config.num_classes)

    def load_panns(self, path: str | Path) -> LoadReport:
        if tuple(self.config.channels) != (64, 128, 256, 512):
            raise ValueError(f"PANNs Cnn10 weights need channels (64, 128, 256, 512), got {self.config.channels}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        result = self.load_state_dict(convert_panns_cnn10(checkpoint), strict=False)
        missing = tuple(key for key in result.missing_keys if not key.startswith("head."))
        report = LoadReport(missing=missing, unexpected=tuple(result.unexpected_keys))
        if not report.is_clean:
            raise ValueError(f"{path}: missing {list(report.missing[:5])}, unexpected {list(report.unexpected[:5])}")
        return report

    def classify(self, x: Tensor) -> Tensor:
        x = x.mean(dim=2)
        x = x.max(dim=2).values + x.mean(dim=2)
        return self.head(self.dropout(self.fc(self.dropout(x)).relu()))

    def forward(self, x: Tensor) -> Tensor:
        return self.classify(self.blocks(x))

    def forward_stages(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        stages = []
        for block in self.blocks:
            x = block(x)
            stages.append(x)
        return self.classify(x), tuple(stages)
