# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path


def rms_normalize(waveform, target_rms: float):
    """
    Chuẩn hóa căn bậc hai giá trị trung bình bình phương (Root Mean Square - RMS) của tín hiệu.
    
    Công thức:
        RMS = sqrt(mean(waveform^2))
        normalized = waveform * (target_rms / RMS)
        
    Mục đích:
        Đưa tất cả các đoạn âm thanh về cùng một mức năng lượng/âm lượng chuẩn (target_rms = 0.07925),
        giúp mô hình nơ-ron không bị ảnh hưởng bởi độ to nhỏ bất thường của thiết bị ghi âm.
    """
    rms = waveform.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    return waveform * (target_rms / rms)


def mirror_file(source: str | Path, backup_root: str | Path, base_dir: str | Path) -> Path:
    """Sao lưu file theo đúng cấu trúc thư mục tương đối tương ứng"""
    source_path = Path(source).resolve()
    base_path = Path(base_dir).resolve()
    try:
        relative = source_path.relative_to(base_path)
    except ValueError:
        relative = Path(source_path.name)
    destination = Path(backup_root).resolve() / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination
