from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ema_beat.config import EMAConfig, EMABeatTrainConfig
from ema_beat.loss import CRDLoss, StudentTeacherEMALoss
from ema_beat.trainer import EMABeatTrainer


def test_crd_loss_initialization():
    crd = CRDLoss(
        student_dim=768,
        teacher_dim=768,
        num_samples=100,
        embed_dim=128,
        negatives=32,
        temperature=0.07,
        momentum=0.5,
    )
    assert crd.bank_student.shape == (100, 128)
    assert crd.bank_teacher.shape == (100, 128)
    assert crd.bank_labels.shape == (100,)
    assert (crd.bank_labels == -1).all()


def test_crd_loss_forward_2d():
    crd = CRDLoss(student_dim=64, teacher_dim=64, num_samples=32, embed_dim=16, negatives=8)
    student_emb = torch.randn(4, 64, requires_grad=True)
    teacher_emb = torch.randn(4, 64)
    index = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([1, 2, 1, 3])

    loss = crd(student_emb, teacher_emb, index=index, labels=labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert student_emb.grad is not None
    assert torch.isfinite(student_emb.grad).all()


def test_crd_loss_forward_3d():
    crd = CRDLoss(student_dim=32, teacher_dim=32, num_samples=16, embed_dim=16, negatives=4)
    # (B, N, D) = (4, 12, 32)
    student_emb = torch.randn(4, 12, 32, requires_grad=True)
    teacher_emb = torch.randn(4, 12, 32)
    loss = crd(student_emb, teacher_emb)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_crd_memory_bank_update():
    crd = CRDLoss(student_dim=16, teacher_dim=16, num_samples=8, embed_dim=8, negatives=4, momentum=0.5)
    student_emb = torch.randn(2, 16)
    teacher_emb = torch.randn(2, 16)
    index = torch.tensor([2, 5])
    labels = torch.tensor([3, 7])

    old_bank_s = crd.bank_student[index].clone()
    assert (crd.bank_labels[index] == -1).all()

    _ = crd(student_emb, teacher_emb, index=index, labels=labels)

    new_bank_s = crd.bank_student[index]
    assert not torch.allclose(old_bank_s, new_bank_s)
    assert (crd.bank_labels[index] == labels).all()


def test_crd_reset_bank():
    crd = CRDLoss(student_dim=16, teacher_dim=16, num_samples=100, embed_dim=16)
    assert crd.bank_student.shape[0] == 100
    crd.reset_bank(16)
    assert crd.bank_student.shape[0] == 16
    assert crd.bank_teacher.shape[0] == 16
    assert crd.bank_labels.shape[0] == 16


def test_student_teacher_loss_with_crd():
    crd = CRDLoss(student_dim=32, teacher_dim=32, num_samples=16, embed_dim=16, negatives=4)
    criterion = StudentTeacherEMALoss(
        kd_weight=1.0,
        temperature=4.0,
        label_smoothing=0.0,
        crd_weight=0.8,
        crd_loss=crd,
    )

    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    student_feats = torch.randn(4, 32, requires_grad=True)
    teacher_feats = torch.randn(4, 32)
    targets = torch.tensor([0, 1, 2, 3])
    index = torch.tensor([0, 1, 2, 3])

    total_loss, details = criterion(
        student_logits=student_logits,
        targets=targets,
        teacher_logits=teacher_logits,
        student_features=student_feats,
        teacher_features=teacher_feats,
        index=index,
    )

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert details["task_loss"] > 0
    assert details["kd_loss"] > 0
    assert details["crd_loss"] > 0

    total_loss.backward()
    assert student_logits.grad is not None
    assert student_feats.grad is not None


class MockModelWithFeatures(nn.Module):
    def __init__(self, in_dim=16, num_classes=4):
        super().__init__()
        self.config = SimpleNamespace(encoder_layers=2, encoder_embed_dim=in_dim, num_classes=num_classes)
        self.encoder = nn.Sequential(nn.Linear(in_dim, in_dim))
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x, **kwargs):
        feat = self.encoder(x[:, :16])
        logits = self.head(feat)
        return SimpleNamespace(logits=logits, features=feat)

    def predict_proba(self, x):
        return torch.softmax(self(x).logits, dim=-1)


class MockPairDataset(Dataset):
    def __init__(self, num_samples=16, dim=16, num_classes=4):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.waveforms = torch.randn(num_samples, dim)
        self.labels = torch.arange(num_samples) % num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        target = torch.zeros(self.num_classes)
        target[self.labels[idx]] = 1.0
        return {
            "waveform": self.waveforms[idx],
            "teacher_waveform": self.waveforms[idx],
            "target": target,
            "index": torch.tensor(idx, dtype=torch.long),
        }


def test_trainer_with_crd(tmp_path: Path):
    loader = DataLoader(MockPairDataset(num_samples=16, dim=16, num_classes=4), batch_size=4)
    train_cfg = EMABeatTrainConfig(epochs=1, batch_size=4, eval_batch_size=4, amp="none")
    ema_cfg = EMAConfig(
        crd_weight=0.8,
        temperature=4.0,
        crd_dim=8,
        crd_negatives=4,
        crd_num_samples=16,
    )

    model = MockModelWithFeatures(in_dim=16, num_classes=4)
    trainer = EMABeatTrainer(
        model=model,
        config=train_cfg,
        ema_config=ema_cfg,
        device=torch.device("cpu"),
        logger=logging.getLogger("test_trainer_crd"),
        name="test_crd_trainer",
        checkpoint_path=tmp_path / "best.pt",
        resume_path=tmp_path / "resume.pt",
    )

    # Check parameter groups include CRD projection parameters
    groups = trainer.get_parameter_groups()
    total_opt_params = sum(len(g["params"]) for g in groups)
    trainable_params = len(trainer.trainable_parameters)
    assert total_opt_params == trainable_params

    # Check CRD parameters are present
    assert any(p is trainer.criterion.crd.embed_student.weight for g in groups for p in g["params"])

    # Run 1 epoch
    result = trainer.fit(loader, loader)
    assert result.epochs_run == 1
    assert trainer.criterion.crd.bank_labels[0].item() != -1
