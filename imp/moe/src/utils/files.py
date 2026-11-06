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
        
    Ghi chú quan trọng:
        Nếu target_rms <= 0 (chế độ bỏ chuẩn hóa - ablation không RMS), hàm sẽ trả về
        dạng sóng thô nguyên bản (raw waveform), KHÔNG nhân với 0.
    """
    if target_rms <= 0.0:
        return waveform
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
