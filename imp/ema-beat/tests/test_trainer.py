from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ema_beat.config import EMAConfig, EMABeatTrainConfig
from ema_beat.trainer import EMABeatTrainer

CLASSES = 4
SAMPLES = 16
DIM = 16


class MockBEATModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=CLASSES)
        self.fc = nn.Linear(DIM, CLASSES)

    def forward(self, waveform: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self.fc(waveform[:, :DIM]))

    def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(waveform).logits, dim=-1)


class MockDataset(Dataset):
    def __init__(self) -> None:
        gen = torch.Generator().manual_seed(42)
        self.waveforms = torch.randn(SAMPLES, DIM, generator=gen)
        self.labels = torch.arange(SAMPLES) % CLASSES

    def __len__(self) -> int:
        return SAMPLES

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        target = torch.zeros(CLASSES)
        target[self.labels[idx]] = 1.0
        return {
            "waveform": self.waveforms[idx],
            "target": target,
            "snr_db": torch.tensor(10.0),
            "sample_id": f"sample_{idx}",
        }


def test_trainer_fit_and_ema_update(tmp_path: Path) -> None:
    loader = DataLoader(MockDataset(), batch_size=4)
    train_cfg = EMABeatTrainConfig(
        epochs=2,
        batch_size=4,
        eval_batch_size=4,
        warmup_ratio=0.1,
        patience=2,
        selection_metric="macro_f1",
        amp="none",
    )
    ema_cfg = EMAConfig(
        decay=0.9,
        warmup_steps=2,
        selection_target="ema",
    )
    logger = logging.getLogger("test_trainer")
    checkpoint_path = tmp_path / "best.pt"
    resume_path = tmp_path / "resume.pt"

    model = MockBEATModel()
    trainer = EMABeatTrainer(
        model=model,
        config=train_cfg,
        ema_config=ema_cfg,
        device=torch.device("cpu"),
        logger=logger,
        name="test_ema_trainer",
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
    )

    # Trước khi fit: shadow weights bằng model
    initial_dist = trainer.ema.weight_distance(trainer.model)
    assert initial_dist == 0.0

    result = trainer.fit(loader, loader)
    assert result.epochs_run == 2

    # Sau khi fit: model đã tối ưu và step, shadow weights đã cập nhật
    assert trainer.ema.step_count > 0
    assert len(trainer.history) == 2
    assert "val_ema_macro_f1" in trainer.history[0]
    assert "val_raw_macro_f1" in trainer.history[0]
    assert "weight_distance" in trainer.history[0]


def test_trainer_save_and_resume(tmp_path: Path) -> None:
    loader = DataLoader(MockDataset(), batch_size=4)
    train_cfg = EMABeatTrainConfig(
        epochs=1,
        batch_size=4,
        eval_batch_size=4,
        selection_metric="macro_f1",
        amp="none",
    )
    ema_cfg = EMAConfig(decay=0.95, warmup_steps=1)
    logger = logging.getLogger("test_resume")
    checkpoint_path = tmp_path / "best.pt"
    resume_path = tmp_path / "resume.pt"

    model = MockBEATModel()
    trainer = EMABeatTrainer(
        model=model,
        config=train_cfg,
        ema_config=ema_cfg,
        device=torch.device("cpu"),
        logger=logger,
        name="test_resume_trainer",
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
    )

    trainer.fit(loader, loader)
    assert resume_path.is_file()

    # Tạo trainer mới và khôi phục từ resume
    new_model = MockBEATModel()
    new_trainer = EMABeatTrainer(
        model=new_model,
        config=train_cfg,
        ema_config=ema_cfg,
        device=torch.device("cpu"),
        logger=logger,
        name="test_resume_trainer_2",
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
    )
    new_trainer.load_resume(resume_path)
    assert new_trainer._completed_epochs == 1
    assert new_trainer.ema.step_count == trainer.ema.step_count


def test_export_inference_model(tmp_path: Path) -> None:
    train_cfg = EMABeatTrainConfig(epochs=1, batch_size=4, eval_batch_size=4, amp="none")
    ema_cfg = EMAConfig(decay=0.99)
    model = MockBEATModel()
    trainer = EMABeatTrainer(
        model=model,
        config=train_cfg,
        ema_config=ema_cfg,
        device=torch.device("cpu"),
        logger=logging.getLogger("test_export"),
        name="test_export",
        checkpoint_path=tmp_path / "best.pt",
    )

    # Đổi tham số model
    with torch.no_grad():
        trainer.model.fc.weight.add_(1.0)
    trainer.ema.update(trainer.model)

    exported = trainer.export_inference_model()
    assert isinstance(exported, nn.Module)
    # Trọng số của exported phải khớp với ema.ema_model
    assert torch.allclose(exported.fc.weight, trainer.ema.ema_model.fc.weight)


def test_layer_decay_parameter_groups(tmp_path: Path) -> None:
    class DummyBEATs(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_classes=CLASSES, encoder_layers=3)
            self.head = SimpleNamespace(fc=nn.Linear(DIM, CLASSES))
            self.backbone = nn.ModuleDict({
                "patch_embed": nn.Linear(DIM, DIM),
                "encoder": nn.ModuleDict({
                    "layers": nn.ModuleList([
                        nn.Linear(DIM, DIM),
                        nn.Linear(DIM, DIM),
                        nn.Linear(DIM, DIM),
                    ])
                })
            })

    model = DummyBEATs()
    train_cfg = EMABeatTrainConfig(lr=1e-4, layer_decay=0.3, epochs=1)
    trainer = EMABeatTrainer(
        model=model,
        config=train_cfg,
        ema_config=EMAConfig(),
        device=torch.device("cpu"),
        logger=logging.getLogger("test_ld"),
        name="test_ld",
        checkpoint_path=tmp_path / "best.pt",
    )
    groups = trainer.get_parameter_groups()
    assert len(groups) == 4  # scale 1.0 (head + layer 2), scale 0.3 (layer 1), scale 0.09 (layer 0), scale 0.027 (stem)
    assert pytest.approx(groups[0]["lr"]) == 1e-4
    assert pytest.approx(groups[1]["lr"]) == 3e-5
    assert pytest.approx(groups[2]["lr"]) == 9e-6
    assert pytest.approx(groups[3]["lr"]) == 2.7e-6


def test_teacher_uses_teacher_waveform(tmp_path: Path) -> None:
    class Recorder(MockBEATModel):
        pass

    model = MockBEATModel()
    trainer = EMABeatTrainer(
        model=model,
        config=EMABeatTrainConfig(epochs=1, batch_size=4, eval_batch_size=4, amp="none"),
        ema_config=EMAConfig(decay=0.99),
        device=torch.device("cpu"),
        logger=logging.getLogger("test_teacher_input"),
        name="test_teacher_input",
        checkpoint_path=tmp_path / "best.pt",
    )
    seen: list[Tensor] = []
    original = trainer.ema.forward

    def spy(x: Tensor, *args: object, **kwargs: object):
        seen.append(x.clone())
        return original(x, *args, **kwargs)

    trainer.ema.forward = spy  # type: ignore[method-assign]

    student_wave = torch.randn(4, DIM)
    clean_wave = torch.randn(4, DIM)
    target = torch.zeros(4, CLASSES)
    target[torch.arange(4), torch.arange(4) % CLASSES] = 1.0
    batch = {"waveform": student_wave, "teacher_waveform": clean_wave, "target": target}
    trainer.training_step(batch)
    assert torch.equal(seen[-1], clean_wave)

    trainer.training_step({"waveform": student_wave, "target": target})
    assert torch.equal(seen[-1], student_wave)


def test_trainer_with_separate_teacher_model(tmp_path: Path) -> None:
    loader = DataLoader(MockDataset(), batch_size=4)
    train_cfg = EMABeatTrainConfig(epochs=1, batch_size=4, eval_batch_size=4, amp="none")
    ema_cfg = EMAConfig(decay=0.999, warmup_steps=0, warmup_decay_start=0.999)
    student = MockBEATModel()
    teacher = MockBEATModel()
    with torch.no_grad():
        teacher.fc.weight.add_(5.0)

    trainer = EMABeatTrainer(
        model=student,
        teacher_model=teacher,
        config=train_cfg,
        ema_config=ema_cfg,
        device=torch.device("cpu"),
        logger=logging.getLogger("test_separate_teacher"),
        name="test_separate_teacher",
        checkpoint_path=tmp_path / "best.pt",
    )
    assert trainer.ema.weight_distance(student) > 0.0
    assert torch.allclose(trainer.ema.ema_model.fc.weight, teacher.fc.weight)
