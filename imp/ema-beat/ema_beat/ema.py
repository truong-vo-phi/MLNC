from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn


def clone_model(model: nn.Module) -> nn.Module:
    """Sao chép một mô hình PyTorch, tự động xử lý trường hợp mô hình có weight_norm
    (như BEATClassifier) khiến copy.deepcopy gặp lỗi RuntimeError.
    """
    try:
        return copy.deepcopy(model)
    except Exception:
        if hasattr(model, "config") and hasattr(model, "__class__"):
            shadow = model.__class__(model.config)
            shadow.load_state_dict(model.state_dict())
            return shadow
        import io

        buf = io.BytesIO()
        torch.save(model, buf)
        buf.seek(0)
        shadow = torch.load(buf, weights_only=False)
        shadow.load_state_dict(model.state_dict())
        return shadow


class BEATsModelEMA(nn.Module):
    """Exponential Moving Average (EMA) cho mô hình BEATs / BEATClassifier.

    Duy trì bản sao shadow weights đã được làm mịn của mô hình BEATs trong suốt
    quá trình huấn luyện.

    Công thức cập nhật:
        theta_ema = beta * theta_ema + (1 - beta) * theta_model

    Các tính năng:
    - Warm-up schedule cho decay: beta tăng dần từ `warmup_decay_start` (ví dụ 0.9)
      lên `decay` (ví dụ 0.9995) qua `warmup_steps` bước, giúp EMA phản ứng nhanh
      với các bước nhảy lớn ở đầu quá trình học.
    - CPU offloading: lưu trữ trọng số EMA trên RAM (CPU) để tiết kiệm VRAM cho GPU
      nhỏ, tự động chuyển sang GPU khi cần dự đoán.
    - Export standalone: xuất mô hình BEATs chuẩn đã được cập nhật trọng số EMA, sẵn
      sàng triển khai suy luận với 0 overhead.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        warmup_steps: int = 1000,
        warmup_decay_start: float = 0.90,
        update_interval: int = 1,
        device: torch.device | None = None,
        cpu_offload: bool = False,
        teacher_model: nn.Module | None = None,
        pretrained_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if not (0.0 <= warmup_decay_start <= decay):
            raise ValueError(
                f"warmup_decay_start ({warmup_decay_start}) must be in [0, decay={decay}]"
            )
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {update_interval}")

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.warmup_decay_start = warmup_decay_start
        self.update_interval = update_interval
        self.cpu_offload = cpu_offload
        self.step_count = 0

        # Khởi tạo Teacher Shadow Model: ưu tiên teacher_model riêng hoặc nạp pretrain
        if teacher_model is not None:
            self.ema_model = clone_model(teacher_model)
        elif pretrained_path is not None:
            self.ema_model = clone_model(model)
            if hasattr(self.ema_model, "load_pretrained"):
                self.ema_model.load_pretrained(pretrained_path)
            elif Path(pretrained_path).is_file():
                ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                self.ema_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        else:
            self.ema_model = clone_model(model)

        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        if self.cpu_offload:
            self.ema_model.to("cpu")
        elif device is not None:
            self.ema_model.to(device)

    def load_pretrained(self, path: str | Path) -> Any:
        """Nạp trọng số pretrained checkpoint vào Teacher EMA shadow model."""
        if hasattr(self.ema_model, "load_pretrained"):
            return self.ema_model.load_pretrained(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return self.ema_model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    def get_decay(self) -> float:
        """Tính toán hệ số suy giảm beta tại bước huấn luyện hiện tại."""
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            progress = self.step_count / self.warmup_steps
            return self.warmup_decay_start + (self.decay - self.warmup_decay_start) * progress
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Cập nhật trọng số shadow model theo trọng số mới nhất của mô hình đang train."""
        self.step_count += 1
        if self.step_count % self.update_interval != 0:
            return

        current_decay = self.get_decay()
        one_minus_decay = 1.0 - current_decay

        # Cập nhật parameters (trọng số và bias)
        source_params = dict(model.named_parameters())
        for name, ema_param in self.ema_model.named_parameters():
            if name in source_params:
                source_p = source_params[name]
                if not source_p.requires_grad:
                    continue
                source_data = source_p.data
                if self.cpu_offload and source_data.device != ema_param.device:
                    source_data = source_data.to(ema_param.device)
                ema_param.data.mul_(current_decay).add_(source_data, alpha=one_minus_decay)

        # Sao chép buffers (LayerNorm hoặc running statistics nếu có)
        source_buffers = dict(model.named_buffers())
        for name, ema_buf in self.ema_model.named_buffers():
            if name in source_buffers:
                source_buf_data = source_buffers[name].data
                if self.cpu_offload and source_buf_data.device != ema_buf.device:
                    source_buf_data = source_buf_data.to(ema_buf.device)
                ema_buf.data.copy_(source_buf_data)

    @torch.no_grad()
    def predict_proba(self, waveform: Tensor) -> Tensor:
        """Dự đoán xác suất các lớp bằng mô hình EMA (tương thích BEATClassifier)."""
        target_device = waveform.device
        if self.cpu_offload and next(self.ema_model.parameters()).device != target_device:
            self.ema_model.to(target_device)
            try:
                if hasattr(self.ema_model, "predict_proba"):
                    return self.ema_model.predict_proba(waveform)
                return torch.softmax(self.ema_model(waveform).logits, dim=-1)
            finally:
                self.ema_model.to("cpu")

        if hasattr(self.ema_model, "predict_proba"):
            return self.ema_model.predict_proba(waveform)
        return torch.softmax(self.ema_model(waveform).logits, dim=-1)

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass thông qua mô hình EMA."""
        return self.ema_model(*args, **kwargs)

    @torch.no_grad()
    def weight_distance(self, model: nn.Module) -> float:
        """Tính khoảng cách L2 (Frobenius norm) giữa trọng số mô hình gốc và shadow EMA:

        ||theta_model - theta_ema||_2.
        """
        total_sq_dist = 0.0
        source_params = dict(model.named_parameters())
        for name, ema_param in self.ema_model.named_parameters():
            if name in source_params and source_params[name].requires_grad:
                diff = source_params[name].data - ema_param.data.to(source_params[name].device)
                total_sq_dist += float(torch.sum(diff * diff).item())
        return math.sqrt(max(0.0, total_sq_dist))

    def export_model(self) -> nn.Module:
        """Xuất một bản sao độc lập của mô hình BEATs chứa trọng số EMA."""
        exported = clone_model(self.ema_model)
        exported.eval()
        return exported

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Lưu trạng thái của module EMA bao gồm cả step_count và shadow weights."""
        return {
            "step_count": self.step_count,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "warmup_decay_start": self.warmup_decay_start,
            "ema_model": self.ema_model.state_dict(*args, **kwargs),
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Khôi phục trạng thái module EMA từ checkpoint."""
        if "ema_model" in state_dict:
            self.step_count = state_dict.get("step_count", 0)
            self.decay = state_dict.get("decay", self.decay)
            self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
            self.warmup_decay_start = state_dict.get(
                "warmup_decay_start", self.warmup_decay_start
            )
            return self.ema_model.load_state_dict(state_dict["ema_model"], strict=strict)
        return self.ema_model.load_state_dict(state_dict, strict=strict)
