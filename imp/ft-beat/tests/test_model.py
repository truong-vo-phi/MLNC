from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ft_beat.model import FRCRNBEATPipeline, rms_normalize
from torch import Tensor, nn

from src.core import BaseModel
from src.model.beat import BEATOutput


class DummySeparator(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DummySeparator:
        return cls()

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return checkpoint

    def forward(self, mixture: Tensor) -> Tensor:
        return self.conv(mixture.unsqueeze(1)).squeeze(1)


class DummyClassifier(BaseModel):
    def __init__(self, num_classes: int = 36) -> None:
        super().__init__()
        self.linear = nn.Linear(1600, num_classes)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DummyClassifier:
        return cls()

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return checkpoint

    def forward(self, waveform: Tensor) -> BEATOutput:
        pooled = waveform[:, :1600]
        logits = self.linear(pooled)
        return BEATOutput(logits=logits, features=pooled, hidden=())

    def predict_proba(self, waveform: Tensor) -> Tensor:
        return self(waveform).logits.softmax(dim=-1)


def test_rms_normalize() -> None:
    waveform = torch.randn(4, 16000)
    norm = rms_normalize(waveform, target_rms=0.07925)
    rms = norm.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.tensor(0.07925), atol=1e-3)


def test_frcrn_beat_pipeline_forward() -> None:
    sep = DummySeparator()
    clf = DummyClassifier(num_classes=36)
    pipeline = FRCRNBEATPipeline(sep, clf, target_rms=0.07925, freeze_separator=True)

    waveform = torch.randn(2, 16000)
    out = pipeline(waveform)
    assert out.logits.shape == (2, 36)
    probs = pipeline.predict_proba(waveform)
    assert probs.shape == (2, 36)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)
