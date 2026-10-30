from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CRDLoss(nn.Module):
    """Contrastive Representation Distillation (Tian et al., 2020) với memory bank cho hai view,
    đối xứng Teacher EMA <-> Student.

    - embed_student: Linear(student_dim, embed_dim) chiếu biểu diễn ẩn của Student
    - embed_teacher: Linear(teacher_dim, embed_dim) chiếu biểu diễn ẩn của Teacher
    - bank_student, bank_teacher: memory banks lưu negative representations
    - bank_labels: lưu nhãn lớp của từng mẫu trong memory bank để lọc false negatives
    """

    def __init__(
        self,
        student_dim: int = 768,
        teacher_dim: int = 768,
        num_samples: int = 30240,
        embed_dim: int = 128,
        negatives: int = 4096,
        temperature: float = 0.07,
        momentum: float = 0.5,
    ) -> None:
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.negatives = negatives
        self.temperature = temperature
        self.momentum = momentum

        self.embed_student = nn.Linear(student_dim, embed_dim)
        self.embed_teacher = nn.Linear(teacher_dim, embed_dim)

        self.register_buffer("bank_student", F.normalize(torch.randn(num_samples, embed_dim), dim=1))
        self.register_buffer("bank_teacher", F.normalize(torch.randn(num_samples, embed_dim), dim=1))
        self.register_buffer("bank_labels", torch.full((num_samples,), -1, dtype=torch.long))

    def reset_bank(self, num_samples: int) -> None:
        """Khởi tạo lại kích thước memory bank khi số lượng mẫu thay đổi (ví dụ: smoke test, test nhỏ)."""
        self.num_samples = num_samples
        device = self.bank_student.device
        self.bank_student = F.normalize(torch.randn(num_samples, self.embed_dim, device=device), dim=1)
        self.bank_teacher = F.normalize(torch.randn(num_samples, self.embed_dim, device=device), dim=1)
        self.bank_labels = torch.full((num_samples,), -1, dtype=torch.long, device=device)

    def _nce(
        self,
        anchor: Tensor,
        positive: Tensor,
        bank: Tensor,
        negative_index: Tensor,
        index: Tensor,
        labels: Tensor,
    ) -> Tensor:
        positive_logit = (anchor * positive).sum(dim=1, keepdim=True)
        negative_logit = torch.einsum("be,bke->bk", anchor, bank[negative_index])
        invalid = (self.bank_labels[negative_index] == labels[:, None]) | (negative_index == index[:, None])
        negative_logit = negative_logit.masked_fill(invalid, float("-inf"))
        logits = torch.cat([positive_logit, negative_logit], dim=1) / self.temperature
        return F.cross_entropy(logits, torch.zeros(len(anchor), dtype=torch.long, device=anchor.device))

    def forward(
        self,
        student_embedding: Tensor,
        teacher_embedding: Tensor,
        index: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Tensor:
        # Nếu biểu diễn ở dạng chuỗi patch token (B, N, D), pool trung bình thành (B, D)
        if student_embedding.ndim == 3:
            student_embedding = student_embedding.mean(dim=1)
        if teacher_embedding.ndim == 3:
            teacher_embedding = teacher_embedding.mean(dim=1)

        batch_size = student_embedding.shape[0]
        device = student_embedding.device

        # Chỉ số mẫu an toàn
        if index is None:
            index = torch.arange(batch_size, device=device) % self.bank_student.shape[0]
        else:
            index = index.to(device)
            if (index >= self.bank_student.shape[0]).any():
                index = index % self.bank_student.shape[0]

        # Nhãn lớp để lọc false negatives
        if labels is None:
            labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)
            else:
                labels = labels.long()

        zs = F.normalize(self.embed_student(student_embedding), dim=1)
        zt = F.normalize(self.embed_teacher(teacher_embedding), dim=1)

        num_bank = self.bank_student.shape[0]
        k_neg = min(self.negatives, num_bank)
        if k_neg < 1:
            return 1.0 - (zs * zt).sum(dim=1).mean()

        negative_index = torch.randint(0, num_bank, (batch_size, k_neg), device=device)
        loss = (
            self._nce(zs, zt, self.bank_teacher, negative_index, index, labels)
            + self._nce(zt, zs, self.bank_student, negative_index, index, labels)
        )

        with torch.no_grad():
            for bank, z in ((self.bank_student, zs), (self.bank_teacher, zt)):
                bank[index] = F.normalize(
                    self.momentum * bank[index] + (1.0 - self.momentum) * z.detach(), dim=1
                )
            self.bank_labels[index] = labels

        return loss


class StudentTeacherEMALoss(nn.Module):
    """Loss function kết hợp Task Loss, Logit Knowledge Distillation (KD) và
    Contrastive Representation Distillation (CRD) giữa Student và Teacher EMA.

    Công thức:
        L_total = L_task + kd_weight * L_kd(T) + crd_weight * L_crd
    """

    def __init__(
        self,
        kd_weight: float = 1.0,
        temperature: float = 4.0,
        label_smoothing: float = 0.0,
        crd_weight: float = 0.8,
        crd_loss: CRDLoss | None = None,
    ) -> None:
        super().__init__()
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.crd_weight = crd_weight
        self.crd = crd_loss

    def forward(
        self,
        student_logits: Tensor,
        targets: Tensor,
        teacher_logits: Tensor | None = None,
        student_features: Tensor | None = None,
        teacher_features: Tensor | None = None,
        index: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        t = self.temperature

        # 1. Task loss
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            target_indices = targets.squeeze().long()
            task_loss = F.cross_entropy(
                student_logits, target_indices, label_smoothing=self.label_smoothing
            )
            is_multilabel = False
        else:
            if self.label_smoothing > 0.0:
                smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                smoothed_targets = targets
            task_loss = F.binary_cross_entropy_with_logits(student_logits, smoothed_targets)
            is_multilabel = True

        # Extract logits if teacher_logits is an output container
        if hasattr(teacher_logits, "logits"):
            teacher_logits = teacher_logits.logits

        # 2. Knowledge distillation loss (Logit KL / BCE)
        if teacher_logits is not None and self.kd_weight > 0.0:
            if is_multilabel:
                teacher_probs = torch.sigmoid(teacher_logits / t)
                kd_loss = F.binary_cross_entropy_with_logits(student_logits / t, teacher_probs) * (t ** 2)
            else:
                student_log_probs = F.log_softmax(student_logits / t, dim=-1)
                teacher_probs = F.softmax(teacher_logits / t, dim=-1)
                kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)
            kd_val = kd_loss.item()
        else:
            kd_loss = torch.tensor(0.0, device=student_logits.device)
            kd_val = 0.0

        # 3. Contrastive Representation Distillation (CRD) loss
        if (
            self.crd is not None
            and self.crd_weight > 0.0
            and student_features is not None
            and teacher_features is not None
        ):
            crd_loss = self.crd(
                student_embedding=student_features,
                teacher_embedding=teacher_features,
                index=index,
                labels=targets,
            )
            crd_val = crd_loss.item()
        else:
            crd_loss = torch.tensor(0.0, device=student_logits.device)
            crd_val = 0.0

        total_loss = task_loss + self.kd_weight * kd_loss + self.crd_weight * crd_loss

        details = {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "kd_loss": kd_val,
            "crd_loss": crd_val,
        }
        return total_loss, details
