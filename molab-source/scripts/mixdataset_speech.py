"""Fine-tune FRCRN trên chính mixture của mix-dataset với tham chiếu là tiếng nói sạch `clean_path`.

Trainer của FRCRN chỉ nhận (mixture, noise) và suy ra tiếng nói là `mixture - noise`. Ở đây đặt `noise = mixture - clean`, nên
với `target = speech` tiếng nói mục tiêu bằng đúng `clean_path`, còn nhiễu tham chiếu là phần dư thật của mixture (không cần
`mixture = clean + noise` chính xác, không cần AudioSet). Train cắt ngẫu nhiên cùng một đoạn của mixture và clean; validation cắt từ đầu.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import AudioItem, Mixture36Dataset, build_dataloader, subsample  # noqa: E402

logger = logging.getLogger(__name__)


class CleanReferenceDataset(Mixture36Dataset):
    """Mixture-36l trả `waveform` (mixture), `noise = mixture - clean`, `snr_db`; mixture và clean luôn cùng đoạn cắt."""

    def __init__(self, *args: Any, random_crop: bool = False, seed: int = 0, **kwargs: Any) -> None:
        kwargs["load_noise"] = False
        kwargs["load_clean"] = False
        super().__init__(*args, **kwargs)
        self.random_crop = random_crop
        self.seed = seed

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        entropy = int(np.random.default_rng().integers(0, 2**63 - 1)) if self.random_crop else None
        mixture = self.read(self.root / row["mixture_path"], None if entropy is None else np.random.default_rng(entropy))
        clean = self.read(self.root / row["clean_path"], None if entropy is None else np.random.default_rng(entropy))
        return {
            "waveform": mixture,
            "noise": mixture - clean,
            "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
            "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


def make_mixdataset_loaders(
    seed: int = 0, train_snr: Sequence[float] | None = None
) -> Callable[[Any, int | None, int], dict[str, DataLoader]]:
    """Hàm dựng loader cho `finetune_frcrn(make_loaders=...)`: nhận (config, max_samples, num_workers).

    train_snr: chỉ huấn luyện trên các mức SNR này (chuyên gia theo dải); validation luôn giữ đủ các mức SNR để so sánh.
    """
    levels = None if train_snr is None else tuple(float(level) for level in train_snr)

    def make(config: Any, max_samples: int | None, num_workers: int) -> dict[str, DataLoader]:
        train = CleanReferenceDataset(
            split="train", duration=config.segment_seconds, random_crop=True, seed=seed, snr_db=levels
        )
        validation = CleanReferenceDataset(split="validation", duration=config.eval_segment_seconds, seed=seed)
        logger.info("mix-dataset (noise = mixture - clean): train %d, validation %d", len(train), len(validation))
        return {
            "train": build_dataloader(subsample(train, max_samples), config.batch_size, True, num_workers),
            "validation": build_dataloader(
                subsample(validation, max_samples), config.eval_batch_size, False, num_workers
            ),
        }

    return make


__all__ = ["CleanReferenceDataset", "make_mixdataset_loaders"]
