from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.model.student import CEDStudent, LogMel, SpecAugment, StudentConfig, load_ced_tiny


def make_student(num_classes: int = 4) -> CEDStudent:
    config = StudentConfig(
        num_classes=num_classes,
        freq_bins=32,
        frames=64,
        init="none",
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
    )
    return CEDStudent(config, LogMel(n_mels=32, num_frames=64), SpecAugment(0, 0, 0, 0))


def test_forward_shape_and_backward() -> None:
    student = make_student()
    student.train()
    wave = torch.randn(2, 16000)
    logits = student(wave).logits
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    loss = logits.logsumexp(dim=-1).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_predict_proba_sums_to_one() -> None:
    student = make_student()
    student.eval()
    probabilities = student.predict_proba(torch.randn(3, 16000))
    assert probabilities.shape == (3, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_parameter_groups_match_total() -> None:
    student = make_student()
    groups = student.describe_params()
    total = sum(parameter.numel() for parameter in student.parameters())
    assert sum(groups.values()) == total


def test_load_ced_rejects_mismatched_embedding_dim(tmp_path: Path) -> None:
    student = make_student()
    path = tmp_path / "ced.safetensors"
    save_file(
        {
            "encoder.norm.weight": torch.zeros(256),
            "encoder.norm.bias": torch.zeros(256),
        },
        path,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_ced_tiny(student.vit, path=str(path))
