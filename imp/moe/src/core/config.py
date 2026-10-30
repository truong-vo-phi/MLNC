# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import yaml

ConfigT = TypeVar("ConfigT", bound="BaseConfig")


@dataclass(frozen=True)
class BaseConfig:
    """
    Lớp cấu hình cơ sở (Base Configuration) bất biến (frozen=True).
    
    Hỗ trợ:
        - Tự động nạp tham số từ từ điển (from_dict) và lọc đúng các trường được khai báo.
        - Nạp trực tiếp từ file YAML (from_yaml) theo section chỉ định (mặc định là 'model').
        - Xuất cấu hình ra định dạng dict (to_dict).
    """
    section: ClassVar[str] = "model"

    @classmethod
    def from_dict(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
        """Khởi tạo đối tượng config từ dictionary, chỉ lấy các key trùng với trường thuộc tính"""
        names = {field.name for field in fields(cls)}
        return cls(
            **{key: tuple(value) if isinstance(value, list) else value for key, value in values.items() if key in names}
        )

    @classmethod
    def from_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        """Đọc và nạp section tương ứng từ file YAML"""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f)[cls.section])

    def to_dict(self) -> dict[str, Any]:
        """Xuất thuộc tính cấu hình thành từ điển Python"""
        return asdict(self)
