from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def group_heads(x: Tensor, heads: int) -> Tensor:
    """(B, H_teacher, ...) -> (B, heads, ...) bằng trung bình các head liền kề."""
    teacher_heads = x.shape[1]
    if teacher_heads % heads:
        raise ValueError(f"{teacher_heads} teacher heads do not group into {heads} student heads")
    return x.reshape(x.shape[0], heads, teacher_heads // heads, *x.shape[2:]).mean(dim=2)


def relation_logits(x: Tensor) -> Tensor:
    """(B, H, T, d) -> x_h x_h^T / sqrt(d), (B, H, T, T), tính ở fp32."""
    x = x.float()
    return x @ x.transpose(-2, -1) / math.sqrt(x.shape[-1])


def relation_kl(student: Tensor, teacher: Tensor) -> Tensor:
    """KL(teacher || student) giữa các ma trận quan hệ theo từng hàng (kiểu MiniLM / QKVLoss của kd-v3)."""
    tokens = student.shape[-1]
    return F.kl_div(
        F.log_softmax(student, dim=-1).reshape(-1, tokens),
        F.softmax(teacher, dim=-1).reshape(-1, tokens),
        reduction="batchmean",
    )


def relation_mse(student: Tensor, teacher: Tensor) -> Tensor:
    """MSE giữa hai ma trận quan hệ đã softmax theo hàng, nhân T để thang giá trị cỡ 1 (đều = 1)."""
    tokens = student.shape[-1]
    return F.mse_loss(F.softmax(student, dim=-1) * tokens, F.softmax(teacher, dim=-1) * tokens)


class AttentionKD(nn.Module):
    """Chưng cất q, k, v và bias của BEATs sang student có số head ít hơn.

    Student: cache mỗi khối có q, k, v (B, H_s, 1 + N, d), position_bias (1, H_s, 1 + N, 1 + N), scale (B, H_s, 1 + N, 1);
    token 0 là CLS và bị bỏ trước khi so. Teacher: cùng khoá nhưng (B, H_t, N, d), không có CLS.
    """

    def __init__(self, qkv_weight: float = 0.0, bias_weight: float = 0.0, pos_weight: float = 0.0, qkv_loss: str = "mse") -> None:
        super().__init__()
        if qkv_loss not in ("mse", "kl"):
            raise ValueError(f"qkv_loss must be 'mse' or 'kl', got {qkv_loss!r}")
        self.qkv_weight, self.bias_weight, self.pos_weight = qkv_weight, bias_weight, pos_weight
        self.qkv_distance = relation_mse if qkv_loss == "mse" else relation_kl

    @property
    def enabled(self) -> bool:
        return self.qkv_weight > 0 or self.bias_weight > 0 or self.pos_weight > 0

    def forward(self, student: Sequence[dict[str, Tensor]], teacher: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        if len(student) != len(teacher):
            raise ValueError(f"{len(student)} student layers vs {len(teacher)} teacher layers")
        zero = next(iter(student[0].values())).new_zeros((), dtype=torch.float32)
        terms = {"qkv": zero, "scale": zero, "pos": zero}
        for s, t in zip(student, teacher, strict=True):
            heads = s["q"].shape[1]
            if self.qkv_weight > 0:
                for name in "qkv":
                    rel_student = relation_logits(s[name][:, :, 1:])
                    with torch.no_grad():
                        rel_teacher = group_heads(relation_logits(t[name]), heads)
                    terms["qkv"] = terms["qkv"] + self.qkv_distance(rel_student, rel_teacher) / 3
            if self.bias_weight > 0 and s["scale"] is not None and t["scale"] is not None:
                with torch.no_grad():
                    target = group_heads(t["scale"].float(), heads)
                terms["scale"] = terms["scale"] + F.mse_loss(s["scale"][:, :, 1:].float(), target)
            if self.pos_weight > 0:
                with torch.no_grad():
                    target = group_heads(t["position_bias"].float(), heads)
                terms["pos"] = terms["pos"] + F.mse_loss(s["position_bias"][:, :, 1:, 1:].float(), target)
        count = len(student)
        return {name: value / count for name, value in terms.items()}

    def total(self, terms: dict[str, Tensor]) -> Tensor:
        return self.qkv_weight * terms["qkv"] + self.bias_weight * terms["scale"] + self.pos_weight * terms["pos"]
