from __future__ import annotations

import torch
from torch import Tensor, nn
try:
    from torchaudio.models import Conformer
except (ImportError, OSError):
    Conformer = object

from .config import BaselineConfig


class ConformerNet(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        reduced_freq = ((config.freq_bins - 3) // 2 + 1 - 3) // 2 + 1
        self.subsample = nn.Sequential(
            nn.Conv2d(1, config.hidden, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden, config.hidden, 3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Linear(config.hidden * reduced_freq, config.hidden)
        self.encoder = Conformer(
            input_dim=config.hidden,
            num_heads=config.heads,
            ffn_dim=config.ffn_ratio * config.hidden,
            num_layers=config.layers,
            depthwise_conv_kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.norm = nn.LayerNorm(config.hidden)
        self.head = nn.Linear(config.hidden, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.subsample(x)
        batch, channels, freq, time = x.shape
        x = self.project(x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq))
        lengths = torch.full((batch,), time, dtype=torch.long, device=x.device)
        x, _ = self.encoder(x, lengths)
        return self.head(self.norm(x.mean(dim=1)))
