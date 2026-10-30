from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml
from torch import Tensor

from src.core import BaseModel

from .config import StudentConfig
from .frontend import LogMel, LogMelConfig, SpecAugment, SpecAugmentConfig
from .vit_bias import ViTSBias, load_ced_tiny


class StudentOutput(NamedTuple):
    """Đầu ra của mô hình Student, chứa logits phân loại 36 lớp [B, 36]"""
    logits: Tensor


class CEDStudent(BaseModel):
    """
    Kiến trúc CEDStudent tích hợp trọn gói End-to-End:
        1. Frontend: Chuyển đổi âm thanh thô (waveform) thành phổ Log-Mel chuẩn hóa [B, 1, 64, 398].
        2. Augmentation: SpecAugment che ngẫu nhiên tần số/thời gian trong khi huấn luyện.
        3. Backbone: ViTSBias (DeiT-Tiny) với cơ chế Gated Relative Position Bias 2D.
        4. Head: Tuyến tính phân loại 36 lớp.
    """
    def __init__(self, config: StudentConfig, frontend: LogMel, spec_augment: SpecAugment) -> None:
        super().__init__()
        self.config = config
        self.frontend = frontend
        self.spec_augment = spec_augment
        self.vit = ViTSBias(
            num_classes=config.num_classes,
            freq_bins=config.freq_bins,
            frames=config.frames,
            pretrained=config.init == "imagenet",
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            drop_path=config.drop_path,
            post_ln=config.post_ln,
            pre_ln=config.pre_ln,
            arch=config.arch,
        )
        self.init_report = self._initialize()

    @classmethod
    def from_yaml(cls, path: str | Path) -> CEDStudent:
        """Khởi tạo toàn bộ mô hình CEDStudent từ file cấu hình YAML"""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        student = StudentConfig.from_dict(raw.get(StudentConfig.section, {}))
        frontend = LogMelConfig.from_dict(raw.get(LogMelConfig.section, {}))
        spec_augment = SpecAugmentConfig.from_dict(raw.get(SpecAugmentConfig.section, {}))
        return cls(student, LogMel(**frontend.to_dict()), SpecAugment(**spec_augment.to_dict()))

    def _initialize(self) -> dict[str, int]:
        """Tải trọng số pretrain ban đầu và khởi tạo zero cho bảng relative position bias"""
        report: dict[str, int] = {}
        if self.config.init == "ced-tiny":
            report = load_ced_tiny(self.vit, self.config.ced_repo, self.config.ced_path)
        if self.config.bias_init == "zero":
            self.vit.attentions[0].relative_attention_bias.weight.data.zero_()
        return report

    def forward(self, wave: Tensor) -> StudentOutput:
        """
        Forward pass end-to-end:
            Waveform [B, T] -> LogMel -> SpecAugment -> ViTSBias -> Logits [B, 36]
            
        Lưu ý kỹ thuật: Frontend LogMel được ép chạy ở float32 (tắt autocast) để tránh sai số
        số học và lỗi tràn số khi tính log và std. ViT backbone sau đó có thể chạy mixed precision.
        """
        with torch.autocast(wave.device.type, enabled=False):
            features = self.frontend(wave.float())
        return StudentOutput(logits=self.vit(self.spec_augment(features)))

    def predict_proba(self, wave: Tensor) -> Tensor:
        """Tính phân phối xác suất Softmax cho âm thanh đầu vào [B, Num_Classes]"""
        return self(wave).logits.softmax(dim=-1)

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        """Chuyển đổi các định dạng checkpoint khác nhau về state_dict chuẩn của PyTorch"""
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def describe_params(self) -> dict[str, int]:
        """Thống kê chi tiết số tham số theo từng thành phần kiến trúc"""
        return self.vit.describe_params()
