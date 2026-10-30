# -*- coding: utf-8 -*-
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from .base import DistillationLoss


class LogitDistillationLoss(DistillationLoss):
    """
    Hàm mất mát chưng cất tri thức (Hinton Knowledge Distillation Loss) dựa trên phân phối Logits.
    
    Nguyên lý:
        - So sánh phân phối xác suất mềm giữa mô hình Học sinh (Student) và Giáo viên (Teacher)
          thông qua độ phân kỳ Kullback-Leibler (KL Divergence).
        - Nhiệt độ T (temperature) làm mềm các giá trị logit, giúp Student nắm bắt được các "dark knowledge"
          (xác suất tương quan giữa các lớp không phải nhãn chính).
        - Nhân với T^2 nhằm giữ cho độ lớn gradient của loss chưng cất luôn tương đương với Cross-Entropy chuẩn
          khi thay đổi nhiệt độ T.
    """
    name = "logit_distillation"

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        """
        Tham số:
            student: Logits dự đoán của Student [B, Num_Classes]
            teacher: Logits dự đoán của Teacher [B, Num_Classes]
        """
        if student.shape != teacher.shape:
            raise ValueError(f"student {tuple(student.shape)} and teacher {tuple(teacher.shape)} must match")
        temperature = self.temperature
        return (
            F.kl_div(
                F.log_softmax(student / temperature, dim=-1),
                F.softmax(teacher / temperature, dim=-1),
                reduction="batchmean",
            )
            * temperature**2
        )
