# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LoadReport:
    """Báo cáo trạng thái sau khi nạp trọng số checkpoint"""
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """True nếu không có key nào bị thiếu hoặc thừa"""
        return not self.missing and not self.unexpected


class BaseModel(nn.Module, ABC):
    """
    Lớp mô hình cơ sở mở rộng từ nn.Module.
    
    Cung cấp các chức năng tiêu chuẩn hóa:
        - Nạp mô hình từ file cấu hình YAML (from_yaml).
        - Đọc checkpoint và chuyển đổi định dạng state_dict linh hoạt (convert_checkpoint).
        - Kiểm tra tính tương thích của trọng số nghiêm ngặt (strict_pretrained).
        - Đóng băng toàn bộ trọng số phục vụ suy luận / chưng cất (freeze).
        - Đếm số lượng tham số huấn luyện hoặc toàn bộ mô hình (num_parameters).
    """
    expected_missing_prefixes: ClassVar[tuple[str, ...]] = ()
    strict_pretrained: ClassVar[bool] = True
    weights_only: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def from_yaml(cls, path: str | Path) -> BaseModel: ...

    @abstractmethod
    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]: ...

    def read_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Đọc file checkpoint từ đĩa về CPU"""
        return torch.load(path, map_location="cpu", weights_only=self.weights_only)

    def load_pretrained(self, path: str | Path) -> LoadReport:
        """Nạp trọng số pretrained và kiểm tra danh sách missing/unexpected keys"""
        checkpoint = self.read_checkpoint(path)
        result = self.load_state_dict(self.convert_checkpoint(checkpoint), strict=False)
        missing = tuple(key for key in result.missing_keys if not key.startswith(self.expected_missing_prefixes))
        report = LoadReport(missing=missing, unexpected=tuple(result.unexpected_keys))
        if self.strict_pretrained and not report.is_clean:
            raise ValueError(f"{path}: missing {list(report.missing[:5])}, unexpected {list(report.unexpected[:5])}")
        return report

    def freeze(self) -> BaseModel:
        """Chuyển mô hình sang chế độ eval và tắt tính gradient (requires_grad = False)"""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Tính tổng số tham số (trainable hoặc all)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
