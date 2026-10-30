from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ft_beat.config import FTBeatTrainConfig
from ft_beat.trainer import FTBeatTrainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.core import BaseModel
from src.model.beat import BEATOutput


@dataclass
class DummyModelConfig:
    num_classes: int = 4


class DummyTrainableClassifier(BaseModel):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.config = DummyModelConfig(num_classes=num_classes)
        self.linear = nn.Linear(32, num_classes)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DummyTrainableClassifier:
        return cls()

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return checkpoint

    def forward(self, waveform: Tensor) -> BEATOutput:
        pooled = waveform[:, :32]
        logits = self.linear(pooled)
        return BEATOutput(logits=logits, features=pooled, hidden=())

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


def test_trainer_fit_and_resume(tmp_path: Path) -> None:
    logger = logging.getLogger("test_trainer")
    device = torch.device("cpu")
    model = DummyTrainableClassifier(num_classes=4)

    config = FTBeatTrainConfig(
        epochs=2,
        batch_size=4,
        accumulation_steps=2,
        eval_batch_size=4,
        warmup_ratio=0.0,
        amp="none",
    )

    ckpt_path = tmp_path / "best.pt"
    resume_path = tmp_path / "resume.pt"

    trainer = FTBeatTrainer(
        model=model,
        config=config,
        device=device,
        logger=logger,
        name="test_trainer",
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

    new_model = DummyTrainableClassifier(num_classes=4)
    new_trainer = FTBeatTrainer(
        model=new_model,
        config=config,
        device=device,
        logger=logger,
        name="test_resume",
        checkpoint_path=tmp_path / "best_resumed.pt",
        separator=None,
        target_rms=0.0,
        resume_path=resume_path,
    )
    new_trainer.load_resume(resume_path)
    assert new_trainer._completed_epochs == 2
