from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from ced_tiny.config import CEDTinyConfig
from ced_tiny.frontend import LogMel, SpecAugment
from ced_tiny.model import CEDTiny, FRCRNCEDPipeline, rms_normalize
from ced_tiny.vit_bias import load_ced_tiny
from safetensors.torch import save_file
from torch import Tensor, nn

from src.core import BaseModel


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


def make_ced(num_classes: int = 4) -> CEDTiny:
    config = CEDTinyConfig(
        num_classes=num_classes,
        freq_bins=32,
        frames=64,
        init="none",
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
    )
    return CEDTiny(config, LogMel(n_mels=32, num_frames=64), SpecAugment(0, 0, 0, 0))


def test_rms_normalize() -> None:
    waveform = torch.randn(4, 16000)
    norm = rms_normalize(waveform, target_rms=0.07925)
    rms = norm.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.tensor(0.07925), atol=1e-3)


def test_ced_tiny_forward_shape_and_backward() -> None:
    ced = make_ced(num_classes=4)
    ced.train()
    wave = torch.randn(2, 16000)
    logits = ced(wave).logits
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    loss = logits.logsumexp(dim=-1).mean()
    loss.backward()
    assert any(p.grad is not None for p in ced.parameters())


def test_ced_tiny_predict_proba() -> None:
    ced = make_ced(num_classes=4)
    ced.eval()
    probs = ced.predict_proba(torch.randn(3, 16000))
    assert probs.shape == (3, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-4)


def test_ced_parameter_groups_match_total() -> None:
    ced = make_ced(num_classes=4)
    groups = ced.describe_params()
    total = sum(p.numel() for p in ced.parameters())
    assert sum(groups.values()) == total


def test_frcrn_ced_pipeline_forward() -> None:
    sep = DummySeparator()
    clf = make_ced(num_classes=4)
    pipeline = FRCRNCEDPipeline(sep, clf, target_rms=0.07925, freeze_separator=True)

    waveform = torch.randn(2, 16000)
    out = pipeline(waveform)
    assert out.logits.shape == (2, 4)
    probs = pipeline.predict_proba(waveform)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_load_ced_rejects_mismatched_embedding_dim(tmp_path: Path) -> None:
    ced = make_ced(num_classes=4)
    path = tmp_path / "ced.safetensors"
    save_file(
        {
            "encoder.norm.weight": torch.zeros(256),
            "encoder.norm.bias": torch.zeros(256),
        },
        path,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_ced_tiny(ced.vit, path=str(path))
