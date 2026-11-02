"""
Knowledge Distillation Losses for Task 3:
Supports:
1. Logit KD (Hinton et al., 2015)
2. CRD (Contrastive Representation Distillation, Tian et al., ICLR 2020)
3. MGD (Masked Generative Distillation, Yang et al., ECCV 2022)
4. logit + CRD
5. logit + MGD
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LogitKDLoss(nn.Module):
    """
    Logit-based Knowledge Distillation Loss (Hinton et al., 2015).
    Computes KL divergence between temperature-scaled student and teacher probabilities.
    L_logit = T^2 * KL( softmax(z_S / T) || softmax(z_T / T) )
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        t = self.temperature
        p_s = F.log_softmax(student_logits / t, dim=-1)
        p_t = F.softmax(teacher_logits / t, dim=-1)
        return self.kl_div(p_s, p_t) * (t * t)


class CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation (Tian et al., ICLR 2020).
    Aligns student and teacher representation geometry using InfoNCE loss.
    Projects student (dim=192) and teacher (dim=768) representations to a shared space (dim=128),
    maximizing agreement between matching pairs while contrasting against negatives in the batch.
    """

    def __init__(
        self,
        student_dim: int = 192,
        teacher_dim: int = 768,
        embed_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.embed_dim = embed_dim

        self.proj_student = nn.Sequential(
            nn.Linear(student_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj_teacher = nn.Sequential(
            nn.Linear(teacher_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _pool(self, feat: Tensor) -> Tensor:
        if feat.ndim == 3:
            # (B, N, D) -> pool over tokens or CLS
            return feat[:, 0] if feat.shape[1] > 1 else feat.squeeze(1)
        return feat

    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        s = self._pool(student_feat)
        t = self._pool(teacher_feat)

        # Project and L2 normalize
        z_s = F.normalize(self.proj_student(s), dim=-1)  # (B, D)
        z_t = F.normalize(self.proj_teacher(t), dim=-1)  # (B, D)

        b = z_s.size(0)
        if b <= 1:
            return torch.tensor(0.0, device=student_feat.device, requires_grad=True)

        # Cosine similarity matrix between student and teacher representations: (B, B)
        sim_matrix = torch.matmul(z_s, z_t.T) / self.temperature

        # Diagonal entries are positive pairs
        labels = torch.arange(b, device=student_feat.device, dtype=torch.long)
        return F.cross_entropy(sim_matrix, labels)


class MGDLoss(nn.Module):
    """
    Masked Generative Distillation (Yang et al., ECCV 2022).
    Disrupts student features using a random binary mask and reconstructs
    the teacher's representations via a lightweight generator network.
    L_MGD = (1 / D) * || Phi(Proj(F_S) * M) - F_T ||_2^2
    """

    def __init__(
        self,
        student_dim: int = 192,
        teacher_dim: int = 768,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.adapt = nn.Linear(student_dim, teacher_dim)
        self.generator = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim),
        )

    def _pool(self, feat: Tensor) -> Tensor:
        if feat.ndim == 3:
            return feat.mean(dim=1)
        return feat

    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        s = self._pool(student_feat)
        t = self._pool(teacher_feat)

        # 1. Adapt student representation to teacher dimension
        s_adapted = self.adapt(s)

        # 2. Generate random binary mask: (B, D)
        if self.training:
            prob = torch.rand_like(s_adapted)
            mask = (prob > self.mask_ratio).float()
            s_masked = s_adapted * mask
        else:
            s_masked = s_adapted

        # 3. Lightweight generator reconstructs teacher representations
        s_rec = self.generator(s_masked)

        # 4. Normalized MSE loss between reconstructed student and teacher feature
        return F.mse_loss(s_rec, t.detach())


class CompositeKDLoss(nn.Module):
    """
    Unified Loss Handler for the 5 Knowledge Distillation methods:
    1. 'logit'     : (1 - alpha)*CE + alpha*L_logit
    2. 'crd'       : CE + beta*L_crd
    3. 'mgd'       : CE + gamma*L_mgd
    4. 'logit+crd' : (1 - alpha)*CE + alpha*L_logit + beta*L_crd
    5. 'logit+mgd' : (1 - alpha)*CE + alpha*L_logit + gamma*L_mgd
    """

    def __init__(
        self,
        method: str = "logit",
        student_dim: int = 192,
        teacher_dim: int = 768,
        num_classes: int = 36,
        temperature: float = 2.0,
        alpha: float = 0.7,
        beta: float = 0.5,
        gamma: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.method = method.lower().replace(" ", "_")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Task loss: Cross Entropy with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # KD loss components
        self.logit_kd = LogitKDLoss(temperature=temperature)
        self.crd_loss = CRDLoss(student_dim=student_dim, teacher_dim=teacher_dim)
        self.mgd_loss = MGDLoss(student_dim=student_dim, teacher_dim=teacher_dim)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_feat: Tensor | None = None,
        teacher_feat: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar loss Tensor for backprop
            loss_dict: dictionary of individual loss components
        """
        # Ground-truth CE loss
        if targets.ndim == 2 and targets.shape[-1] > 1:
            target_indices = targets.argmax(dim=-1)
        else:
            target_indices = targets.long()
        loss_ce = self.ce_loss(student_logits, target_indices)

        loss_dict = {"ce": float(loss_ce.item())}
        total_loss = loss_ce

        # 1. Logit KD
        if self.method in ("logit", "logit+crd", "logit_crd", "logit+mgd", "logit_mgd"):
            l_logit = self.logit_kd(student_logits, teacher_logits)
            loss_dict["logit_kd"] = float(l_logit.item())
            total_loss = (1.0 - self.alpha) * total_loss + self.alpha * l_logit

        # 2. CRD Loss
        if self.method in ("crd", "logit+crd", "logit_crd"):
            if student_feat is not None and teacher_feat is not None:
                l_crd = self.crd_loss(student_feat, teacher_feat)
                loss_dict["crd"] = float(l_crd.item())
                total_loss = total_loss + self.beta * l_crd

        # 3. MGD Loss
        if self.method in ("mgd", "logit+mgd", "logit_mgd"):
            if student_feat is not None and teacher_feat is not None:
                l_mgd = self.mgd_loss(student_feat, teacher_feat)
                loss_dict["mgd"] = float(l_mgd.item())
                total_loss = total_loss + self.gamma * l_mgd

        loss_dict["total"] = float(total_loss.item())
        return total_loss, loss_dict

