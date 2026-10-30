from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from ban_beat.config import BANTeacherConfig
from ban_beat.teacher_finetune import train_teacher
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.beat import BEATClassifier, BEATConfig
from src.train import BEATTrainConfig

CLASSES = 5
SAMPLES = 8


class FrozenSeparator(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform * 0.5

    def freeze(self) -> FrozenSeparator:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class WaveDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.waveforms = torch.randn(SAMPLES, 16000, generator=generator) * 0.1
        self.labels = torch.arange(SAMPLES) % CLASSES

    def __len__(self) -> int:
        return SAMPLES

    def __getitem__(self, index: int) -> dict:
        target = torch.zeros(CLASSES)
        target[self.labels[index]] = 1.0
        return {"waveform": self.waveforms[index], "target": target, "snr_db": 0.0, "sample_id": f"s{index}"}


def small_classifier() -> BEATClassifier:
    config = BEATConfig(
        embed_dim=16,
        encoder_embed_dim=32,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_embed_dim=64,
        num_classes=CLASSES,
    )
    return BEATClassifier(config)


def test_finetune_requires_frcrn_checkpoint() -> None:
    with pytest.raises(ValueError, match="frcrn_checkpoint"):
        BANTeacherConfig(finetune=True)


def test_train_teacher_runs_one_epoch_and_saves_best(tmp_path: Path) -> None:
    loader = DataLoader(WaveDataset(), batch_size=4)
    train_config = BEATTrainConfig(epochs=1, patience=1, batch_size=4, eval_batch_size=4, warmup_ratio=0.0, lr=1e-3)
    checkpoint_path = tmp_path / "teacher" / "best.pt"
    result = train_teacher(
        small_classifier(),
        FrozenSeparator(),
        train_config,
        {"train": loader, "validation": loader},
        torch.device("cpu"),
        logging.getLogger("test-teacher"),
        checkpoint_path,
        tmp_path / "resume" / "teacher.pt",
    )
    assert result.best_epoch == 1
    assert checkpoint_path.is_file()


def test_restore_teacher_classifier_loads_saved_weights(tmp_path: Path) -> None:
    from ban_beat.teacher_finetune import restore_teacher_classifier

    from src.utils.checkpoint import save_checkpoint

    trained = small_classifier()
    path = save_checkpoint(tmp_path / "teacher" / "last_best_checkpoint.pt", trained, epoch=7)
    restored = small_classifier()
    epoch = restore_teacher_classifier(restored, path)
    assert epoch == 7
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
