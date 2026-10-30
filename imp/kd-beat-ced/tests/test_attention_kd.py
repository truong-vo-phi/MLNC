from __future__ import annotations

import pytest
import torch

from src.loss.attention_kd import (
    AttentionKDConfig,
    BeatAttentionKD,
    attention_transfer_loss,
    bias_losses,
    group_heads,
    pool_frequency,
)

TIME, TEACHER_FREQ, STUDENT_FREQ = 24, 8, 4
BATCH, CHANNELS, HEADS_T, HEADS_S = 2, 16, 12, 3
STUDENT_TOKENS = TIME * STUDENT_FREQ


def config(**overrides) -> AttentionKDConfig:
    values = {"time_patches": TIME, "teacher_freq_patches": TEACHER_FREQ, "student_freq_patches": STUDENT_FREQ}
    values.update(overrides)
    return AttentionKDConfig(**values)


def expand_index() -> torch.Tensor:
    return torch.tensor([t * STUDENT_FREQ + f // 2 for t in range(TIME) for f in range(TEACHER_FREQ)])


def test_pool_frequency_pairs_adjacent_bins_in_time_major_order() -> None:
    tokens = torch.arange(TIME * TEACHER_FREQ, dtype=torch.float32).reshape(1, -1)
    pooled = pool_frequency(tokens, 1, TIME, TEACHER_FREQ, STUDENT_FREQ)
    expected = torch.tensor([[t * TEACHER_FREQ + f for f in (0, 2, 4, 6)] for t in range(TIME)], dtype=torch.float32)
    expected = ((expected + expected + 1) / 2).reshape(1, -1)
    torch.testing.assert_close(pooled, expected)


def test_group_heads_averages_consecutive_heads() -> None:
    x = torch.arange(HEADS_T, dtype=torch.float32).reshape(1, HEADS_T, 1, 1)
    grouped = group_heads(x, HEADS_S)
    torch.testing.assert_close(grouped.flatten(), torch.tensor([1.5, 5.5, 9.5]))


def test_group_heads_rejects_uneven_grouping() -> None:
    with pytest.raises(ValueError, match="do not group"):
        group_heads(torch.zeros(1, 12, 2), 5)


def test_attention_transfer_is_zero_when_student_matches_pooled_teacher() -> None:
    student = torch.randn(BATCH, STUDENT_TOKENS, CHANNELS)
    teacher = student[:, expand_index()]
    assert attention_transfer_loss((student,), (teacher,), config()).item() == pytest.approx(0.0, abs=1e-6)


def test_attention_transfer_is_positive_for_mismatched_maps() -> None:
    student = torch.randn(BATCH, STUDENT_TOKENS, CHANNELS)
    teacher = torch.randn(BATCH, TIME * TEACHER_FREQ, CHANNELS)
    assert attention_transfer_loss((student,), (teacher,), config()).item() > 0.0


def test_attention_transfer_does_not_pass_gradient_to_teacher() -> None:
    student = torch.randn(BATCH, STUDENT_TOKENS, CHANNELS, requires_grad=True)
    teacher = torch.randn(BATCH, TIME * TEACHER_FREQ, CHANNELS, requires_grad=True)
    attention_transfer_loss((student,), (teacher,), config()).backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_bias_is_zero_when_teacher_is_the_expanded_student_bias() -> None:
    student_scale = torch.rand(BATCH, HEADS_S, STUDENT_TOKENS, 1) + 0.5
    student_pos = torch.randn(1, HEADS_S, STUDENT_TOKENS, STUDENT_TOKENS)
    index = expand_index()
    teacher_scale = student_scale[:, :, index].repeat_interleave(HEADS_T // HEADS_S, dim=1)
    teacher_pos = student_pos[:, :, index][:, :, :, index].repeat_interleave(HEADS_T // HEADS_S, dim=1)
    student_cache = {
        "q": torch.zeros(BATCH, HEADS_S, STUDENT_TOKENS + 1, 4),
        "scale": torch.cat([torch.zeros(BATCH, HEADS_S, 1, 1), student_scale], dim=2),
        "position_bias": torch.cat(
            [
                torch.zeros(1, HEADS_S, 1, STUDENT_TOKENS + 1),
                torch.cat([torch.zeros(1, HEADS_S, STUDENT_TOKENS, 1), student_pos], dim=3),
            ],
            dim=2,
        ),
    }
    teacher_cache = {"scale": teacher_scale, "position_bias": teacher_pos}
    scale, pos = bias_losses([student_cache], [teacher_cache], config())
    assert scale.item() == pytest.approx(0.0, abs=1e-6)
    assert pos.item() == pytest.approx(0.0, abs=1e-6)


def test_module_returns_zero_total_when_all_terms_are_off() -> None:
    module = BeatAttentionKD(AttentionKDConfig(at_weight=0.0, bias_scale_weight=0.0, bias_pos_weight=0.0))
    terms = module([], [], [], [])
    assert terms["total"].item() == 0.0


def test_config_rejects_invalid_frequency_grouping() -> None:
    with pytest.raises(ValueError, match="multiple"):
        AttentionKDConfig(teacher_freq_patches=8, student_freq_patches=3)


def test_config_rejects_layers_outside_beats() -> None:
    with pytest.raises(ValueError, match=r"in 1\.\.12"):
        AttentionKDConfig(at_layers=(13,))
