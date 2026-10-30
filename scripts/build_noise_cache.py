"""Tính n^ = FRCRN(mixture) cho Mixture-36l (4 giây) và ghi cache theo định dạng của build_cached_loaders.

    python -m scripts.build_noise_cache --smoke
    python -m scripts.build_noise_cache --checkpoint pretrained/FRCRN/best_e037_20261020T203958Z.pt

Đầu ra: <out-dir>/<split>.npy (float32, shape (số clip, 64000), đúng thứ tự manifest) và <split>.done (số clip).
n^ chưa chuẩn hoá RMS; SharedFront chuẩn hoá khi huấn luyện teacher và student.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY, SAMPLE_RATE, build_dataloader, build_dataset  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402

FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"
DEFAULT_CHECKPOINT = ROOT / "pretrained" / "FRCRN" / "best_e037_20261020T203958Z.pt"
DEFAULT_OUT_DIR = ROOT / "cache" / "nhat"
SPLITS = ("train", "validation")
DURATION = 4.0
NUM_WORKERS = 4

logger = logging.getLogger("build_noise_cache")


def load_separator(checkpoint: Path, device: torch.device) -> FRCRNSeparator:
    separator = FRCRNSeparator.from_yaml(FRCRN_CONFIG)
    report = separator.load_pretrained(checkpoint)
    logger.info("FRCRN %s: missing=%d, unexpected=%d", checkpoint.name, len(report.missing), len(report.unexpected))
    return separator.to(device).freeze()


@torch.no_grad()
def build_split(
    split: str,
    separator: FRCRNSeparator,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    limit: int | None,
    start_row: int = 0,
) -> None:
    dataset = build_dataset(MIXTURE_KEY, split, duration=DURATION)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count < len(dataset):
        dataset = Subset(dataset, range(count))
    if start_row:
        dataset = Subset(dataset, range(start_row, count))
    loader = build_dataloader(dataset, batch_size, False, NUM_WORKERS)
    samples = int(DURATION * SAMPLE_RATE)

    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"{split}.done"
    path = out_dir / f"{split}.npy"
    if start_row == 0:
        done.unlink(missing_ok=True)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(count, samples))
    else:
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != (count, samples):
            raise RuntimeError(f"{path}: shape {array.shape}, expected {(count, samples)}")
    offset, started = start_row, time.time()
    for batch in tqdm(loader, desc=f"nhat {split}", unit="batch", dynamic_ncols=True):
        estimate = separator(batch["waveform"].to(device)).float().cpu().numpy()
        array[offset : offset + len(estimate)] = estimate
        offset += len(estimate)
    array.flush()
    if offset != count:
        raise RuntimeError(f"{split}: wrote {offset} rows, expected {count}")
    done.write_text(str(count), encoding="utf-8")
    logger.info("%s: %d clip in %.1f s -> %s", split, count, time.time() - started, out_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="chỉ tính n clip đầu mỗi split")
    parser.add_argument("--smoke", action="store_true", help="8 clip mỗi split, ghi vào output/test/nhat-smoke")
    parser.add_argument("--start-row", type=int, default=0, help="tiếp tục từ dòng này (mở file có sẵn, không tạo lại)")
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    args = parser.parse_args(argv)

    if args.smoke:
        args.limit = 8
        args.out_dir = ROOT / "output" / "test" / "nhat-smoke"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "mixture-36l: %s | device %s | out %s", ensure_dataset(MIXTURE_KEY, args.download_root), device, args.out_dir
    )

    separator = load_separator(args.checkpoint, device)
    for split in args.splits:
        build_split(split, separator, device, args.out_dir, args.batch_size, args.limit, args.start_row)


if __name__ == "__main__":
    main()
