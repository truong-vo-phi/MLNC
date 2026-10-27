"""Trộn động cho FRCRN: tiếng nói `clean_path` của mix-dataset + nhiễu AudioSet 36 lớp (bỏ 4 nhãn của 40L).

AudioSet 40L có 4 nhãn không nằm trong 36 lớp của Mixture-36l (đọc từ `labels.txt`). Clip nào mang một trong 4 nhãn đó
bị loại khỏi kho nhiễu. Mixture và nhiễu tham chiếu đều dựng từ `mix_at_snr` nên `mixture = speech + noise` chính xác
(không cần khôi phục `noise_path`). Train rút ngẫu nhiên mỗi lần; validation cố định theo `seed + chỉ mục`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import (  # noqa: E402
    LABELS_FALLBACK,
    MIXTURE_KEY,
    AudioSet40Dataset,
    DynamicMixDataset,
    build_dataloader,
    resolve_dataset_root,
    subsample,
)

logger = logging.getLogger(__name__)


def mixture_class_names(root: str | Path | None = None) -> list[str]:
    root = Path(root) if root else resolve_dataset_root(MIXTURE_KEY)
    return [line.strip() for line in (root / "labels.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def dropped_label_names(class_names: Sequence[str], labels_file: Path = LABELS_FALLBACK) -> list[str]:
    """Nhãn của AudioSet 40L không nằm trong `class_names` (36 lớp)."""
    import csv

    with open(labels_file, encoding="utf-8-sig", newline="") as f:
        names = [row["name"] for row in csv.DictReader(f)]
    kept = set(class_names)
    return [name for name in names if name not in kept]


def filter_noise_36l(noise: AudioSet40Dataset, dropped: Sequence[str]) -> int:
    """Bỏ clip mang nhãn bị loại khỏi `noise.samples`; trả số clip đã bỏ."""
    indices = [noise.class_names.index(name) for name in dropped]
    before = len(noise.samples)
    noise.samples = [(path, target) for path, target in noise.samples if not bool(target[indices].any())]
    return before - len(noise.samples)


def dynamic_36l_dataset(
    split: str, duration: float, snr_levels: Sequence[float], *, deterministic: bool, seed: int = 0
) -> DynamicMixDataset:
    class_names = mixture_class_names()
    dropped = dropped_label_names(class_names)
    noise = AudioSet40Dataset(split=split, duration=None)
    removed = filter_noise_36l(noise, dropped)
    logger.info("%s: nhiễu AudioSet 36L: bỏ %d clip mang nhãn %s, còn %d", split, removed, dropped, len(noise.samples))
    if not noise.samples:
        raise RuntimeError(f"{split}: không còn clip nhiễu nào sau khi bỏ nhãn {dropped}")
    return DynamicMixDataset(
        split=split, noise_dataset=noise, snr_levels=snr_levels, duration=duration, deterministic=deterministic, seed=seed
    )


def make_dynamic_36l_loaders(seed: int = 0) -> Callable[[Any, int | None, int], dict[str, DataLoader]]:
    """Hàm dựng loader cho `finetune_frcrn(make_loaders=...)`: nhận (config, max_samples, num_workers)."""

    def make(config: Any, max_samples: int | None, num_workers: int) -> dict[str, DataLoader]:
        train = dynamic_36l_dataset("train", config.segment_seconds, config.snr_levels, deterministic=False, seed=seed)
        validation = dynamic_36l_dataset(
            "validation", config.eval_segment_seconds, config.snr_levels, deterministic=True, seed=seed
        )
        logger.info("dynamic 36L: train %d giọng nói, validation %d", len(train), len(validation))
        return {
            "train": build_dataloader(subsample(train, max_samples), config.batch_size, True, num_workers),
            "validation": build_dataloader(
                subsample(validation, max_samples), config.eval_batch_size, False, num_workers
            ),
        }

    return make


__all__ = ["dynamic_36l_dataset", "dropped_label_names", "filter_noise_36l", "make_dynamic_36l_loaders"]
