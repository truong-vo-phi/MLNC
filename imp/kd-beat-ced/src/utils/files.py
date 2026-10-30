from __future__ import annotations

import shutil
from pathlib import Path


def rms_normalize(waveform, target_rms: float):
    rms = waveform.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    return waveform * (target_rms / rms)


def mirror_file(source: str | Path, backup_root: str | Path, base_dir: str | Path) -> Path:
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
