"""Tham chiếu nhiễu dựng từ clip AudioSet gốc cho Mixture-36l (thay cho `noise_path`).

`noise_path` của mix-dataset là một đoạn 4 s của clip AudioSet (mono, 16 kHz) nhân `noise_scale * post_gain`.
Vị trí đoạn cắt không được ghi trong manifest, nên bước `offsets` tìm nó một lần bằng tương quan chéo và ghi kèm sai số
khôi phục. `noise_path` chỉ dùng ở bước này để định vị và đo sai số; khi huấn luyện, tham chiếu nhiễu đọc từ clip AudioSet.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import (  # noqa: E402
    AUDIOSET_KEY,
    MIXTURE_KEY,
    SAMPLE_RATE,
    AudioItem,
    Mixture36Dataset,
    build_dataloader,
    fit_length,
    read_audio,
    read_csv,
    resolve_dataset_root,
    subsample,
)

logger = logging.getLogger(__name__)

OFFSET_FIELDS = ("sample_id", "noise_ytid", "offset", "error")
SPLITS = ("train", "validation", "test")


def clip_cache_path(cache_dir: str | Path, ytid: str) -> Path:
    return Path(cache_dir) / f"{ytid}.npy"


def cache_clip(ytid: str, audioset_root: str | Path, cache_dir: str | Path) -> Path:
    """Clip AudioSet -> mono 16 kHz float32 (cùng đường xử lý `read_audio` của bộ nạp), lưu `.npy` để mmap."""
    target = clip_cache_path(cache_dir, ytid)
    if not target.is_file():
        waveform = read_audio(Path(audioset_root) / "audio" / f"{ytid}.wav", SAMPLE_RATE)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{ytid}.{os.getpid()}.tmp.npy")
        np.save(partial, waveform.astype(np.float32))
        os.replace(partial, target)  # nguyên tử: worker khác không bao giờ đọc file dở
    return target


def _cache_job(args: tuple[str, str, str]) -> str:
    ytid, audioset_root, cache_dir = args
    cache_clip(ytid, audioset_root, cache_dir)
    return ytid


def build_clip_cache(ytids: Iterable[str], audioset_root: str | Path, cache_dir: str | Path, workers: int = 4) -> int:
    jobs = [(ytid, str(audioset_root), str(cache_dir)) for ytid in sorted(set(ytids))]
    if workers <= 1:
        done = sum(1 for _ in map(_cache_job, jobs))
    else:
        with ProcessPoolExecutor(workers) as pool:
            done = sum(1 for _ in pool.map(_cache_job, jobs, chunksize=16))
    logger.info("clip cache: %d clip -> %s", done, cache_dir)
    return done


def locate_segment(clip: np.ndarray, reference: np.ndarray, gain: float) -> tuple[int, float]:
    """Vị trí đoạn của `clip` khớp nhất với `reference` và sai số năng lượng tương đối của `gain * đoạn`."""
    length = len(reference)
    if len(clip) < length:
        clip = np.tile(clip, int(np.ceil(length / len(clip))))
    clip = clip.astype(np.float64)
    reference = reference.astype(np.float64)
    correlation = fftconvolve(clip, reference[::-1], mode="valid")
    cumulative = np.concatenate(([0.0], np.cumsum(clip**2)))
    energy = np.sqrt(cumulative[length:] - cumulative[:-length] + 1e-30)
    offset = int(np.argmax(np.abs(correlation) / energy))
    residual = reference - gain * clip[offset : offset + length]
    return offset, float(np.sum(residual**2) / (np.sum(reference**2) + 1e-30))


def _locate_job(args: tuple[str, str, str, str, float, float]) -> tuple[str, str, int, float]:
    sample_id, ytid, noise_path, cache_dir, scale, post_gain = args
    reference, _ = sf.read(noise_path, dtype="float32")
    clip = np.load(clip_cache_path(cache_dir, ytid), mmap_mode="r")
    offset, error = locate_segment(np.asarray(clip), reference, scale * post_gain)
    return sample_id, ytid, offset, error


def compute_offsets(
    split: str,
    out_csv: str | Path,
    cache_dir: str | Path,
    root: str | Path | None = None,
    audioset_root: str | Path | None = None,
    workers: int = 4,
    limit: int | None = None,
) -> int:
    """Ghi bảng (sample_id, noise_ytid, offset, error) của một split; tự dựng clip cache nếu thiếu."""
    root = Path(root) if root else resolve_dataset_root(MIXTURE_KEY)
    audioset_root = Path(audioset_root) if audioset_root else resolve_dataset_root(AUDIOSET_KEY)
    rows = read_csv(root / split / "manifest.csv")
    if limit is not None:
        rows = rows[:: max(len(rows) // limit, 1)][:limit]
    build_clip_cache((row["noise_ytid"] for row in rows), audioset_root, cache_dir, workers)
    jobs = [
        (
            row["sample_id"],
            row["noise_ytid"],
            str(root / row["noise_path"]),
            str(cache_dir),
            float(row["noise_scale"]),
            float(row["post_gain"]),
        )
        for row in rows
    ]
    with ProcessPoolExecutor(max(workers, 1)) as pool:
        results = list(pool.map(_locate_job, jobs, chunksize=32))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OFFSET_FIELDS)
        writer.writerows(results)
    logger.info("offsets %s: %d hàng -> %s", split, len(results), out_csv)
    return len(results)


class AudioSetNoiseMixDataset(Mixture36Dataset):
    """Mixture-36l: `waveform` là mixture của mix-dataset, `noise` dựng từ clip AudioSet gốc (không đọc `noise_path`).

    Chỉ giữ hàng có sai số khôi phục <= `max_error` trong bảng offsets. Mixture và nhiễu cắt cùng vị trí đầu đoạn.
    """

    def __init__(
        self,
        offsets_path: str | Path,
        cache_dir: str | Path,
        max_error: float,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float = 4.0,
        audioset_root: str | Path | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration, load_noise=False, load_clean=False)
        self.audioset_root = Path(audioset_root) if audioset_root else None
        table = {row["sample_id"]: (int(row["offset"]), float(row["error"])) for row in read_csv(Path(offsets_path))}
        self.total_rows = len(self.rows)
        self.rows = [row for row in self.rows if row["sample_id"] in table and table[row["sample_id"]][1] <= max_error]
        self.offsets = [table[row["sample_id"]][0] for row in self.rows]
        self.cache_dir = Path(cache_dir)

    def __getitem__(self, index: int) -> AudioItem:
        item = super().__getitem__(index)
        row = self.rows[index]
        start, length = self.offsets[index], int(self.num_samples)
        cached = clip_cache_path(self.cache_dir, row["noise_ytid"])
        if not cached.is_file():  # dựng clip lần đầu cần tới; nên chạy được cả khi chưa có cache đầy đủ
            self.audioset_root = self.audioset_root or resolve_dataset_root(AUDIOSET_KEY)
            cache_clip(row["noise_ytid"], self.audioset_root, self.cache_dir)
        clip = np.load(cached, mmap_mode="r")
        segment = fit_length(np.asarray(clip[start : start + length], dtype=np.float32), length)
        gain = np.float32(float(row["noise_scale"]) * float(row["post_gain"]))
        item["noise"] = torch.from_numpy(np.ascontiguousarray(segment * gain, dtype=np.float32))
        return item


def audioset_noise_loaders(
    batch_size: int,
    eval_batch_size: int,
    *,
    train_seconds: float,
    eval_seconds: float,
    offsets_dir: str | Path,
    cache_dir: str | Path,
    max_error: float,
    max_samples: int | None = None,
    num_workers: int = 4,
    audioset_root: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Loaders train/validation (cùng khoá với `mixture_loaders` của `scripts.finetune`); không dùng test."""

    def build(split: str, seconds: float, size: int, shuffle: bool) -> DataLoader:
        dataset = AudioSetNoiseMixDataset(
            Path(offsets_dir) / f"offsets_{split}.csv",
            cache_dir,
            max_error,
            split=split,
            duration=seconds,
            audioset_root=audioset_root,
        )
        logger.info("%s: giữ %d / %d hàng (sai số <= %g)", split, len(dataset), dataset.total_rows, max_error)
        return build_dataloader(subsample(dataset, max_samples), size, shuffle, num_workers)

    return {
        "train": build("train", train_seconds, batch_size, True),
        "validation": build("validation", eval_seconds, eval_batch_size, False),
    }


def make_audioset_noise_loaders(
    offsets_dir: str | Path, cache_dir: str | Path, max_error: float
) -> Callable[[Any, int | None, int], dict[str, DataLoader]]:
    """Hàm dựng loader cho `finetune_frcrn(make_loaders=...)`: nhận (config, max_samples, num_workers)."""

    def make(config: Any, max_samples: int | None, num_workers: int) -> dict[str, DataLoader]:
        return audioset_noise_loaders(
            config.batch_size,
            config.eval_batch_size,
            train_seconds=config.segment_seconds,
            eval_seconds=config.eval_segment_seconds,
            offsets_dir=offsets_dir,
            cache_dir=cache_dir,
            max_error=max_error,
            max_samples=max_samples,
            num_workers=num_workers,
        )

    return make


def summarize_errors(csv_path: str | Path, thresholds: Sequence[float] = (1e-3, 1e-2, 5e-2, 1e-1)) -> dict[str, Any]:
    errors = np.array([float(row["error"]) for row in read_csv(Path(csv_path))])
    return {
        "rows": int(len(errors)),
        "median": float(np.median(errors)),
        **{f"kept@{t:g}": float((errors <= t).mean()) for t in thresholds},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    offsets = sub.add_parser("offsets", help="tính bảng offset và sai số cho các split")
    offsets.add_argument("--splits", nargs="+", default=["train", "validation"], choices=SPLITS)
    offsets.add_argument("--out-dir", type=Path, required=True)
    offsets.add_argument("--cache-dir", type=Path, required=True)
    offsets.add_argument("--workers", type=int, default=8)
    offsets.add_argument("--limit", type=int, default=None, help="chỉ lấy ~N hàng mỗi split (thử nhanh)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for split in args.splits:
        csv_path = args.out_dir / f"offsets_{split}.csv"
        compute_offsets(split, csv_path, args.cache_dir, workers=args.workers, limit=args.limit)
        logger.info("%s: %s", split, summarize_errors(csv_path))


if __name__ == "__main__":
    main()
