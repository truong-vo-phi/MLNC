from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core import BaseConfig

NUM_BEAT_LAYERS = 12


@dataclass(frozen=True)
class AttentionKDConfig(BaseConfig):
    section: ClassVar[str] = "attention"
    at_weight: float = 0.0
    at_layers: tuple[int, ...] = (4, 8, 12)
    bias_scale_weight: float = 0.0
    bias_pos_weight: float = 0.0
    bias_layers: tuple[int, ...] = tuple(range(1, NUM_BEAT_LAYERS + 1))
    time_patches: int = 24
    teacher_freq_patches: int = 8
    student_freq_patches: int = 4

    def __post_init__(self) -> None:
        for name in ("at_weight", "bias_scale_weight", "bias_pos_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("at_layers", "bias_layers"):
            layers = getattr(self, name)
            if any(not 1 <= layer <= NUM_BEAT_LAYERS for layer in layers):
                raise ValueError(f"{name} must be in 1..{NUM_BEAT_LAYERS}, got {layers}")
        if self.teacher_freq_patches % self.student_freq_patches:
            raise ValueError("teacher_freq_patches must be a multiple of student_freq_patches")
        if self.time_patches < 1:
            raise ValueError("time_patches must be positive")

    @property
    def at_enabled(self) -> bool:
        return self.at_weight > 0 and len(self.at_layers) > 0

    @property
    def bias_enabled(self) -> bool:
        return (self.bias_scale_weight > 0 or self.bias_pos_weight > 0) and len(self.bias_layers) > 0


def group_heads(x: Tensor, heads: int) -> Tensor:
    teacher_heads = x.shape[1]
    if teacher_heads % heads:
        raise ValueError(f"{teacher_heads} teacher heads do not group into {heads} student heads")
    return x.reshape(x.shape[0], heads, teacher_heads // heads, *x.shape[2:]).mean(dim=2)


def pool_frequency(x: Tensor, dim: int, time: int, teacher_freq: int, student_freq: int) -> Tensor:
    x = x.movedim(dim, 1)
    rest = x.shape[2:]
    x = x.reshape(x.shape[0], time, teacher_freq, *rest)
    x = x.reshape(x.shape[0], time, student_freq, teacher_freq // student_freq, *rest).mean(dim=3)
    x = x.reshape(x.shape[0], time * student_freq, *rest)
    return x.movedim(1, dim)


def token_energy(hidden: Tensor) -> Tensor:
    return hidden.float().pow(2).mean(dim=-1)


def attention_transfer_loss(
    student_hidden: Sequence[Tensor], teacher_hidden: Sequence[Tensor], config: AttentionKDConfig
) -> Tensor:
    if len(student_hidden) != len(teacher_hidden):
        raise ValueError(f"student has {len(student_hidden)} layers, teacher has {len(teacher_hidden)}")
    losses = []
    for student, teacher in zip(student_hidden, teacher_hidden, strict=True):
        with torch.no_grad():
            teacher_map = pool_frequency(
                token_energy(teacher), 1, config.time_patches, config.teacher_freq_patches, config.student_freq_patches
            )
            teacher_map = F.normalize(teacher_map, dim=-1)
        student_map = F.normalize(token_energy(student), dim=-1)
        if student_map.shape != teacher_map.shape:
            raise ValueError(
                f"student tokens {tuple(student_map.shape)} and pooled teacher {tuple(teacher_map.shape)} differ"
            )
        losses.append((student_map - teacher_map).pow(2).mean())
    return torch.stack(losses).mean()


def bias_losses(
    student_caches: Sequence[dict[str, Any]], teacher_caches: Sequence[dict[str, Any]], config: AttentionKDConfig
) -> tuple[Tensor, Tensor]:
    if len(student_caches) != len(teacher_caches):
        raise ValueError(f"student has {len(student_caches)} layers, teacher has {len(teacher_caches)}")
    scale_terms, pos_terms = [], []
    for s, t in zip(student_caches, teacher_caches, strict=True):
        heads = s["q"].shape[1]
        with torch.no_grad():
            teacher_scale = pool_frequency(
                group_heads(t["scale"].float(), heads),
                2,
                config.time_patches,
                config.teacher_freq_patches,
                config.student_freq_patches,
            )
            teacher_pos = group_heads(t["position_bias"].float(), heads)
            teacher_pos = pool_frequency(
                teacher_pos, 2, config.time_patches, config.teacher_freq_patches, config.student_freq_patches
            )
            teacher_pos = pool_frequency(
                teacher_pos, 3, config.time_patches, config.teacher_freq_patches, config.student_freq_patches
            )
        scale_terms.append(F.mse_loss(s["scale"][:, :, 1:].float(), teacher_scale))
        pos_terms.append(F.mse_loss(s["position_bias"][:, :, 1:, 1:].float(), teacher_pos))
    return torch.stack(scale_terms).mean(), torch.stack(pos_terms).mean()


class BeatAttentionKD(nn.Module):
    def __init__(self, config: AttentionKDConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        student_hidden: Sequence[Tensor],
        teacher_hidden: Sequence[Tensor],
        student_caches: Sequence[dict[str, Any]],
        teacher_caches: Sequence[dict[str, Any]],
    ) -> dict[str, Tensor]:
        zero = torch.zeros((), device=self._device(student_hidden, student_caches))
        terms = {"at": zero, "bias_scale": zero, "bias_pos": zero}
        if self.config.at_enabled:
            terms["at"] = attention_transfer_loss(student_hidden, teacher_hidden, self.config)
        if self.config.bias_enabled:
            terms["bias_scale"], terms["bias_pos"] = bias_losses(student_caches, teacher_caches, self.config)
        terms["total"] = (
            self.config.at_weight * terms["at"]
            + self.config.bias_scale_weight * terms["bias_scale"]
            + self.config.bias_pos_weight * terms["bias_pos"]
        )
        return terms

    @staticmethod
    def _device(student_hidden: Sequence[Tensor], student_caches: Sequence[dict[str, Any]]) -> torch.device:
        if student_hidden:
            return student_hidden[0].device
        if student_caches:
            return student_caches[0]["q"].device
        return torch.device("cpu")
