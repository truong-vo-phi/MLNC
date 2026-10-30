from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from ced_tiny.config import CEDTrainConfig
from ced_tiny.loss import SoftTargetCrossEntropy
from ced_tiny.model import CEDOutput
from ced_tiny.trainer import CEDTinyTrainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.core import BaseModel


@dataclass
class DummyModelConfig:
    num_classes: int = 4


class DummyTrainableCED(BaseModel):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.config = DummyModelConfig(num_classes=num_classes)
        self.linear = nn.Linear(32, num_classes)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DummyTrainableCED:
        return cls()

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return checkpoint

    def forward(self, waveform: Tensor) -> CEDOutput:
        pooled = waveform[:, :32]
        logits = self.linear(pooled)
        return CEDOutput(logits=logits)

    def predict_proba(self, waveform: Tensor) -> Tensor:
        return self(waveform).logits.softmax(dim=-1)


class DummyWaveformDataset(Dataset):
    def __init__(self, size: int = 16, num_classes: int = 4) -> None:
        self.size = size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        target = torch.zeros(self.num_classes)
        target[index % self.num_classes] = 1.0
        return {
            "waveform": torch.randn(64),
            "target": target,
            "snr_db": torch.tensor(10.0),
            "sample_id": f"s_{index}",
        }


def test_soft_target_cross_entropy() -> None:
    loss_fn = SoftTargetCrossEntropy(label_smoothing=0.1)
    preds = torch.randn(4, 5)
    hard_targets = torch.tensor([0, 1, 2, 3])
    loss_hard = loss_fn(preds, hard_targets)
    assert torch.isfinite(loss_hard)

    soft_targets = torch.eye(5)[:4]
    loss_soft = loss_fn(preds, soft_targets)
    assert torch.isfinite(loss_soft)

    with pytest.raises(ValueError, match="label_smoothing must be in"):
        SoftTargetCrossEntropy(label_smoothing=1.0)


def test_trainer_fit_and_resume(tmp_path: Path) -> None:
    logger = logging.getLogger("test_trainer")
    device = torch.device("cpu")
    model = DummyTrainableCED(num_classes=4)

    config = CEDTrainConfig(
        epochs=2,
        batch_size=4,
        accumulation_steps=1,
        eval_batch_size=4,
        warmup_ratio=0.0,
        mixup=0.0,
        amp="none",
    )

    ckpt_path = tmp_path / "best.pt"
    resume_path = tmp_path / "resume.pt"

    trainer = CEDTinyTrainer(
        model=model,
        config=config,
        device=device,
        logger=logger,
        name="test_ced_trainer",
        checkpoint_path=ckpt_path,
        separator=None,
        target_rms=0.0,
        resume_path=resume_path,
    )

    train_loader = DataLoader(DummyWaveformDataset(16), batch_size=4)
    val_loader = DataLoader(DummyWaveformDataset(8), batch_size=4)

    res = trainer.fit(train_loader, val_loader)
    assert res.epochs_run == 2
    assert ckpt_path.is_file()
    assert resume_path.is_file()

    trainer_resumed = CEDTinyTrainer(
        model=DummyTrainableCED(num_classes=4),
        config=config,
        device=device,
        logger=logger,
        name="test_ced_trainer_resume",
        checkpoint_path=ckpt_path,
        separator=None,
        target_rms=0.0,
        resume_path=resume_path,
    )
    trainer_resumed.configure_optimizer(steps_per_epoch=4)
    assert trainer_resumed._completed_epochs == 2
