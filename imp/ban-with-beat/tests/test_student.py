from __future__ import annotations

import torch
from ban_beat.student import BANStudent
from ban_beat.teacher import BANTeacher
from torch import nn

from src.model.beat import BEATClassifier, BEATConfig


class HalfSeparator(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform * 0.5


def small_classifier(seed: int) -> BEATClassifier:
    torch.manual_seed(seed)
    config = BEATConfig(
        embed_dim=16,
        encoder_embed_dim=32,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_embed_dim=64,
        num_classes=5,
    )
    return BEATClassifier(config)


def test_student_matches_teacher_with_same_weights() -> None:
    separator = HalfSeparator()
    teacher_classifier = small_classifier(0)
    student_classifier = small_classifier(1)
    student_classifier.load_state_dict(teacher_classifier.state_dict())
    teacher = BANTeacher(teacher_classifier, separator, input_rms=0.05).freeze()
    student = BANStudent(student_classifier, separator, input_rms=0.05).eval()
    waveform = torch.randn(2, 16000) * 0.1
    with torch.no_grad():
        torch.testing.assert_close(student(waveform).logits, teacher(waveform), atol=1e-5, rtol=1e-5)


def test_student_separator_stays_frozen_in_train_mode() -> None:
    separator = HalfSeparator()
    student = BANStudent(small_classifier(0), separator, input_rms=0.05)
    student.train()
    assert not separator.training
    assert all(not parameter.requires_grad for parameter in separator.parameters())
    assert all(parameter.requires_grad for parameter in student.classifier.parameters())


def test_student_without_separator_uses_waveform_directly() -> None:
    student = BANStudent(small_classifier(0), separator=None, input_rms=None)
    waveform = torch.randn(2, 16000) * 0.1
    assert student(waveform).logits.shape == (2, 5)


def test_student_returns_requested_hidden_layers() -> None:
    student = BANStudent(small_classifier(0), separator=None, input_rms=None).eval()
    output = student(torch.randn(2, 16000) * 0.1, layers=(1,))
    assert len(output.hidden) == 1
    assert output.hidden[0].shape[0] == 2 and output.hidden[0].shape[-1] == 32


def test_teacher_hidden_matches_student_hidden_with_same_weights() -> None:
    separator = HalfSeparator()
    teacher_classifier = small_classifier(0)
    student_classifier = small_classifier(1)
    student_classifier.load_state_dict(teacher_classifier.state_dict())
    teacher = BANTeacher(teacher_classifier, separator, input_rms=0.05).freeze()
    student = BANStudent(student_classifier, separator, input_rms=0.05).eval()
    waveform = torch.randn(2, 16000) * 0.1
    _, teacher_hidden = teacher.forward_with_hidden(waveform, (1,))
    with torch.no_grad():
        student_hidden = student(waveform, layers=(1,)).hidden
    torch.testing.assert_close(student_hidden[0], teacher_hidden[0], atol=1e-5, rtol=1e-5)


def test_as_teacher_reproduces_student_logits() -> None:
    student = BANStudent(small_classifier(0), HalfSeparator(), input_rms=0.05).eval()
    teacher = student.as_teacher()
    waveform = torch.randn(2, 16000) * 0.1
    with torch.no_grad():
        torch.testing.assert_close(teacher(waveform), student(waveform).logits)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
