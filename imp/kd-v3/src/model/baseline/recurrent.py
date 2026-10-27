from __future__ import annotations

from torch import Tensor, nn

from .config import BaselineConfig
from .conv_block import conv_stack


class CRNN(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.blocks = conv_stack(config.channels, config.dropout)
        self.gru = nn.GRU(
            config.channels[-1],
            config.hidden,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(2 * config.hidden, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x).mean(dim=2).transpose(1, 2)
        x, _ = self.gru(x)
        return self.head(self.dropout(x.mean(dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.time_pool = nn.AvgPool1d(config.time_pool)
        self.lstm = nn.LSTM(
            config.freq_bins,
            config.hidden,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(2 * config.hidden, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.time_pool(x.squeeze(1)).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(self.dropout(x.mean(dim=1)))
