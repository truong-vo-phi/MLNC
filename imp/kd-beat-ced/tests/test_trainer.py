from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.student import CEDStudent, LogMel, SpecAugment, StudentConfig
from src.train import KDLossConfig, KDTrainConfig, KDTrainer


class ToyTeacher(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, wave: torch.Tensor):
        pooled = wave.mean(dim=-1, keepdim=True)
        return SimpleNamespace(logits=self.fc(pooled))


class ToyDataset(Dataset):
    def __init__(self, length: int = 4, num_classes: int = 4) -> None:
        self.length = length
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        target = torch.zeros(self.num_classes)
        target[index % self.num_classes] = 1.0
        return {
            "waveform": torch.randn(16000, generator=generator) * 0.1,
            "target": target,
            "snr_db": torch.tensor(float((index % 3) * 5), dtype=torch.float32),
            "sample_id": f"toy_{index}",
        }


def make_student() -> CEDStudent:
    config = StudentConfig(
        num_classes=4,
        freq_bins=32,
        frames=64,
        init="none",
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
    )
    return CEDStudent(config, LogMel(n_mels=32, num_frames=64), SpecAugment(0, 0, 0, 0))


def make_config(epochs: int = 1) -> KDTrainConfig:
    return KDTrainConfig(
        epochs=epochs,
        patience=3,
        lr=1.0e-3,
        weight_decay=0.0,
        warmup_steps=1,
        batch_size=2,
        eval_batch_size=2,
        mixup=0.8,
        train_eval_clips=4,
        num_workers=0,
        amp=False,
        selection_metric="score",
    )


def make_trainer(student: CEDStudent, tmp_path: Path, config: KDTrainConfig, resume: bool = False) -> KDTrainer:
    loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
    return KDTrainer(
        student,
        ToyTeacher(),
        config,
        KDLossConfig(temperature=2.0, weight=1.0, label_smoothing=0.1),
        torch.device("cpu"),
        logging.getLogger("test-kd-v5"),
        "toy",
        tmp_path / "best.pt",
        tmp_path / "state.pt",
        1.0,
        train_eval_loader=loader,
        resume=resume,
    )


def test_fit_writes_checkpoint_and_history(tmp_path: Path) -> None:
    student = make_student()
    trainer = make_trainer(student, tmp_path, make_config())
    train_loader = DataLoader(ToyDataset(), batch_size=2, shuffle=True)
    val_loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
    result = trainer.fit(train_loader, val_loader)
    assert result.epochs_run == 1
    assert result.best_epoch == 1
    assert len(result.history) == 1
    row = result.history[0]
    assert torch.isfinite(torch.tensor(row["train_loss"]))
    assert 0.0 <= row["val_macro_acc"] <= 1.0
    assert "val_score" in row
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "state.pt").is_file()
    metrics = trainer.evaluate(val_loader)
    assert {"macro_acc", "mAP", "score"} <= set(metrics)


def test_resume_continues_from_saved_epoch(tmp_path: Path) -> None:
    first = make_trainer(make_student(), tmp_path, make_config(epochs=1))
    train_loader = DataLoader(ToyDataset(), batch_size=2, shuffle=True)
    val_loader = DataLoader(ToyDataset(), batch_size=2, shuffle=False)
    first_result = first.fit(train_loader, val_loader)
    second = make_trainer(make_student(), tmp_path, make_config(epochs=2), resume=True)
    second_result = second.fit(train_loader, val_loader)
    assert len(second_result.history) == 2
    assert second_result.history[0]["epoch"] == 1
    assert second_result.history[0]["val_score"] == pytest.approx(first_result.history[0]["val_score"])
    assert second_result.history[1]["epoch"] == 2
    assert second_result.epochs_run == 2
