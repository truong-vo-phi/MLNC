from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core import BaseConfig

from .base import DistillationLoss, Loss, SupervisedLoss
from .cross_entropy import CrossEntropyLoss

MEMBERSHIP_FILE = Path(__file__).resolve().parents[2] / "config" / "membership.npy"
PROB_EPS = 1e-8


@dataclass(frozen=True)
class KDMoELossConfig(BaseConfig):
    section: ClassVar[str] = "loss"

    label_smoothing: float = 0.1
    kd_temperature: float = 2.0
    kd_weight: float = 1.0
    kd_weight_by_snr: bool = False
    route_ce_weight: float = 0.0
    front_weight: float = 1.0
    front_lam_over: float = 2.0
    cons_weight: float = 0.1


def load_membership(path: Path = MEMBERSHIP_FILE) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"membership matrix not found, run scripts.build_group_map first: {path}")
    membership = torch.from_numpy(np.load(path)).float()
    if membership.ndim != 2:
        raise ValueError(f"membership must be 2D [classes, groups], got {tuple(membership.shape)}")
    return membership


def teacher_group_probs(teacher_logits: Tensor, membership: Tensor, temperature: float) -> Tensor:
    probs = (teacher_logits.detach().float() / temperature).softmax(dim=-1)
    return probs @ membership.to(probs.device).float()


def router_kl(
    router_logits: Tensor,
    teacher_logits: Tensor,
    membership: Tensor,
    temperature: float,
    weights: Tensor | None = None,
) -> Tensor:
    p_t = teacher_group_probs(teacher_logits, membership, temperature)
    log_p_s = (router_logits.float() / temperature).log_softmax(dim=-1)
    kl = (p_t * (p_t.clamp_min(PROB_EPS).log() - log_p_s)).sum(dim=-1)
    if weights is not None:
        kl = kl * weights.to(kl.device).float()
    return (temperature * temperature) * kl


class RouterKDLoss(DistillationLoss):
    """KL nhiệt độ giữa phân phối nhóm của router và của teacher, gom từ logit 36 lớp."""

    name = "router_kd"

    def __init__(self, membership: Tensor, temperature: float = 2.0) -> None:
        super().__init__()
        if membership.ndim != 2:
            raise ValueError(f"membership must be 2D [classes, groups], got {tuple(membership.shape)}")
        self.register_buffer("membership", membership.detach().clone().float())
        self.temperature = temperature

    def forward(self, student: Tensor, teacher: Tensor, weights: Tensor | None = None) -> Tensor:
        kl = router_kl(student, teacher, self.membership, self.temperature, weights)
        return kl.mean()


class FrontLoss(SupervisedLoss):
    """L1 bất đối xứng trên log-mel nhiễu: phạt phần ước lượng thừa (rò tiếng nói) nặng hơn."""

    name = "front"

    def __init__(self, lam_over: float = 2.0) -> None:
        super().__init__()
        self.lam_over = lam_over

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction.float() - target.float()
        weight = 1.0 + (self.lam_over - 1.0) * (diff > 0).float()
        return (weight * diff.abs()).mean()


class ConsistencyLoss(Loss):
    """1 - cos trên các cặp mix cùng clip nhiễu trong batch, gom theo pair_id."""

    name = "cons"

    def forward(self, embedding: Tensor, pair_ids: Sequence[str] | Tensor) -> Tensor:
        normalized = F.normalize(embedding.float(), dim=-1)
        if isinstance(pair_ids, Tensor):
            keys = pair_ids
        else:
            lookup = {value: index for index, value in enumerate(sorted(set(pair_ids)))}
            keys = torch.tensor([lookup[value] for value in pair_ids])
        losses = []
        for key in keys.unique():
            members = (keys == key).nonzero(as_tuple=True)[0]
            if members.numel() < 2:
                continue
            group = normalized[members]
            similarity = group @ group.T
            rows, cols = torch.triu_indices(members.numel(), members.numel(), offset=1)
            losses.append((1.0 - similarity[rows, cols]).mean())
        if not losses:
            return normalized.new_zeros(())
        return torch.stack(losses).mean()


class KDMoELoss(nn.Module):
    """Tổng loss: CE + KL router + L_front + L_cons (+ CE nhãn nhóm), mọi số hạng ở fp32."""

    name = "kdmoe"

    def __init__(self, config: KDMoELossConfig, membership: Tensor) -> None:
        super().__init__()
        self.config = config
        if membership.ndim != 2:
            raise ValueError(f"membership must be 2D [classes, groups], got {tuple(membership.shape)}")
        self.register_buffer("membership", membership.detach().clone().float())
        self.ce = CrossEntropyLoss(config.label_smoothing)
        self.kd = RouterKDLoss(membership, config.kd_temperature)
        self.front = FrontLoss(config.front_lam_over)
        self.cons = ConsistencyLoss()
        self.snr_accuracy: Mapping[float, float] | None = None

    def _snr_weights(self, snr_db: Tensor) -> Tensor | None:
        if not self.config.kd_weight_by_snr or not self.snr_accuracy:
            return None
        best = max(self.snr_accuracy.values())
        if best <= 0.0:
            return None
        values = torch.as_tensor(
            [self.snr_accuracy.get(round(float(snr), 1), best) for snr in snr_db.tolist()], dtype=torch.float32
        )
        return values / best

    def forward(self, output: Any, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        terms: dict[str, Tensor] = {}
        labels = batch["label"]
        terms["ce"] = self.ce(output.logits, labels)
        weights = self._snr_weights(batch["snr_db"]) if "snr_db" in batch else None
        terms["kd"] = self.kd(output.router_logits, batch["teacher_logits36"], weights)
        if output.noise_mel is not None and output.noise_target is not None:
            terms["front"] = self.front(output.noise_mel, output.noise_target)
        terms["cons"] = self.cons(output.e, batch["noise_ytid"])
        if self.config.route_ce_weight:
            terms["route_ce"] = F.cross_entropy(output.router_logits.float(), batch["group"])
        total = terms["ce"] + self.config.kd_weight * terms["kd"] + self.config.cons_weight * terms["cons"]
        if "front" in terms:
            total = total + self.config.front_weight * terms["front"]
        if "route_ce" in terms:
            total = total + self.config.route_ce_weight * terms["route_ce"]
        terms["total"] = total
        return terms
