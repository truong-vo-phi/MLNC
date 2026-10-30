from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ban_beat.config import BANLossConfig, BANTrainConfig
from ban_beat.loss import attention_transfer_loss
from ban_beat.trainer import BANTrainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

CLASSES = 3
SAMPLES = 16
DIM = 8


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=CLASSES)
        self.linear = nn.Linear(DIM, CLASSES)

    def forward(self, waveform: torch.Tensor, layers: tuple[int, ...] = ()) -> SimpleNamespace:
        return SimpleNamespace(logits=self.linear(waveform[:, :DIM]), hidden=())

    def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
        return self(waveform).logits.softmax(dim=-1)


class TinyDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.waveforms = torch.randn(SAMPLES, DIM, generator=generator)
        self.labels = torch.arange(SAMPLES) % CLASSES
        self.snrs = torch.tensor([-5.0, 5.0, 15.0, 20.0] * (SAMPLES // 4))

    def __len__(self) -> int:
        return SAMPLES

    def __getitem__(self, index: int) -> dict:
        target = torch.zeros(CLASSES)
        target[self.labels[index]] = 1.0
        return {
            "waveform": self.waveforms[index],
            "target": target,
            "snr_db": self.snrs[index],
            "sample_id": f"sample-{index}",
        }


def test_trainer_runs_one_epoch(tmp_path: Path) -> None:
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=4)
    config = BANTrainConfig(epochs=1, patience=1, batch_size=4, eval_batch_size=4, warmup_ratio=0.0, lr=1e-2)
    logits = {f"sample-{index}": torch.randn(CLASSES) for index in range(SAMPLES)}
    trainer = BANTrainer(
        model=TinyModel(),
        config=config,
        loss_config=BANLossConfig(kd_weight=0.5, temperature=1.0),
        snr_levels=(-5.0, 5.0, 15.0, 20.0),
        teacher=None,
        teacher_logits=logits,
        device=torch.device("cpu"),
        logger=logging.getLogger("test"),
        name="tiny",
        checkpoint_path=tmp_path / "tiny-best.pt",
        resume_path=tmp_path / "tiny-resume.pt",
    )
    result = trainer.fit(loader, loader)
    assert result.best_epoch == 1
    assert "val_macro_f1_snr" in result.history[0]
    assert trainer.history[0]["val_macro_f1_snr"] == result.history[0]["val_macro_f1_snr"]
    assert result.best_score == result.history[0]["val_macro_f1_snr"]
    assert (tmp_path / "tiny-best.pt").is_file()


HIDDEN_TOKENS = 4


class HiddenModel(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=CLASSES)
        self.linear = nn.Linear(DIM, CLASSES)
        self.scale = scale

    def hidden(self, waveform: torch.Tensor, layers: tuple[int, ...]) -> tuple[Tensor, ...]:
        base = waveform[:, :DIM].unsqueeze(1).expand(-1, HIDDEN_TOKENS, -1)
        return tuple(base * self.scale * layer for layer in layers)

    def forward(self, waveform: torch.Tensor, layers: tuple[int, ...] = ()) -> SimpleNamespace:
        return SimpleNamespace(logits=self.linear(waveform[:, :DIM]), hidden=self.hidden(waveform, layers))


class HiddenTeacher(HiddenModel):
    def forward(self, waveform: torch.Tensor) -> Tensor:
        return self.linear(waveform[:, :DIM]).detach()

    def forward_with_hidden(self, waveform: torch.Tensor, layers: tuple[int, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
        return self.linear(waveform[:, :DIM]).detach(), tuple(h.detach() for h in self.hidden(waveform, layers))


def test_attention_step_uses_teacher_hidden_and_adds_loss(tmp_path: Path) -> None:
    dataset = TinyDataset()
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    teacher = HiddenTeacher(scale=2.0)
    student = HiddenModel(scale=1.0)
    config = BANTrainConfig(epochs=1, patience=1, batch_size=4, eval_batch_size=4, warmup_ratio=0.0, lr=1e-2)
    kwargs = {
        "config": config,
        "snr_levels": (-5.0, 5.0, 15.0, 20.0),
        "device": torch.device("cpu"),
        "logger": logging.getLogger("test"),
        "name": "tiny",
        "checkpoint_path": tmp_path / "tiny-best.pt",
    }
    with_attention = BANTrainer(
        model=student,
        loss_config=BANLossConfig(kd_weight=0.5, attention_weight=1000.0, attention_layers=(1, 2)),
        teacher=teacher,
        teacher_logits=None,
        **kwargs,
    )
    without_attention = BANTrainer(
        model=student,
        loss_config=BANLossConfig(kd_weight=0.5, attention_weight=0.0),
        teacher=teacher,
        teacher_logits=None,
        **kwargs,
    )
    with_attention.model.train()
    without_attention.model.train()
    loss_with = with_attention.training_step(batch).loss
    loss_without = without_attention.training_step(batch).loss
    assert torch.isfinite(loss_with)
    waveform = batch["waveform"]
    expected = 1000.0 / 2 * attention_transfer_loss(student.hidden(waveform, (1, 2)), teacher.hidden(waveform, (1, 2)))
    torch.testing.assert_close(loss_with - loss_without, expected)
    loss_with.backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert with_attention.model.linear.weight.grad is not None


def test_attention_requires_teacher(tmp_path: Path) -> None:
    config = BANTrainConfig(epochs=1, patience=1, batch_size=4, eval_batch_size=4, warmup_ratio=0.0, lr=1e-2)
    trainer = BANTrainer(
        model=HiddenModel(scale=1.0),
        config=config,
        loss_config=BANLossConfig(kd_weight=0.5, attention_weight=1.0, attention_layers=(1,)),
        snr_levels=(-5.0,),
        teacher=None,
        teacher_logits=None,
        device=torch.device("cpu"),
        logger=logging.getLogger("test"),
        name="tiny",
        checkpoint_path=tmp_path / "tiny-best.pt",
    )
    batch = next(iter(DataLoader(TinyDataset(), batch_size=4)))
    with pytest.raises(ValueError, match="teacher model is required"):
        trainer.training_step(batch)


def test_attention_transfer_loss_is_zero_for_identical_hidden() -> None:
    hidden = (torch.randn(2, HIDDEN_TOKENS, DIM),)
    assert attention_transfer_loss(hidden, hidden).item() == 0.0


def test_plateau_reduces_lr_every_patience_epochs(tmp_path: Path) -> None:
    config = BANTrainConfig(
        epochs=1,
        patience=5,
        batch_size=4,
        eval_batch_size=4,
        warmup_ratio=0.0,
        lr=1e-2,
        plateau_factor=0.5,
        plateau_patience=2,
    )
    trainer = BANTrainer(
        model=TinyModel(),
        config=config,
        loss_config=BANLossConfig(kd_weight=0.0),
        snr_levels=(-5.0,),
        teacher=None,
        teacher_logits=None,
        device=torch.device("cpu"),
        logger=logging.getLogger("test"),
        name="tiny",
        checkpoint_path=tmp_path / "tiny-best.pt",
    )
    trainer.configure_optimizer(steps_per_epoch=2)
    before = list(trainer.scheduler.base_lrs)
    trainer.on_no_improvement(1, 1)
    assert trainer.scheduler.base_lrs == before
    trainer.on_no_improvement(2, 2)
    assert trainer.scheduler.base_lrs == [lr * 0.5 for lr in before]
