"""Đo SI-SNR của FRCRN theo từng mức SNR đầu vào (chỉ suy luận), cùng công thức với `FRCRNTrainer.evaluate`.

Tham chiếu nhiễu là `mixture - clean` (như lúc fine-tune trên mix-dataset). Báo cho từng checkpoint:
  noise_si_snr_<mức>  : SI-SNR của nhiễu ước lượng n_hat = mixture - tiếng nói ước lượng (dB)
  speech_si_snr_<mức> : SI-SNR của tiếng nói ước lượng (dB)
  *_hi                : trung bình hai mức 15 và 20 dB (tiêu chí chọn checkpoint)

    python -m scripts.eval_frcrn_snr --frcrn pretrained/FRCRN/best_e024_20261020T145757Z.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY, build_dataloader  # noqa: E402
from scripts.mixdataset_speech import CleanReferenceDataset  # noqa: E402
from src.loss.frcrn import si_snr  # noqa: E402
from src.metrics import SnrBucketMeter  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402

LEVELS = (-5, 0, 5, 10, 15, 20)
logger = logging.getLogger("eval_frcrn_snr")


@torch.no_grad()
def frcrn_snr_table(loader: Any, separator: Callable[[Tensor], Tensor], device: torch.device, desc: str = "eval") -> dict[str, float]:
    noise_meter, speech_meter = SnrBucketMeter(), SnrBucketMeter()
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        mixture, noise = batch["waveform"].to(device), batch["noise"].to(device)
        estimate = separator(mixture)
        noise_meter.update(si_snr(noise, estimate), batch["snr_db"])
        speech_meter.update(si_snr(mixture - noise, mixture - estimate), batch["snr_db"])
    return {**noise_meter.compute("noise_si_snr"), **speech_meter.compute("speech_si_snr")}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frcrn", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "eval-frcrn-snr.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger.info("mixture-36l: %s", ensure_dataset(MIXTURE_KEY, args.download_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {}
    for path in args.frcrn:
        separator = FRCRNSeparator.from_yaml(ROOT / "config" / "frcrn.yaml")
        separator.load_pretrained(path)
        separator.to(device).freeze()
        report[str(path)] = {}
        for split in args.splits:
            dataset = CleanReferenceDataset(split=split, duration=args.duration)
            scores = frcrn_snr_table(build_dataloader(dataset, args.batch_size, False, args.workers), separator, device, f"{path.name} {split}")
            report[str(path)][split] = scores
            line = " ".join(f"{level}:{scores[f'noise_si_snr_{level}']:.2f}" for level in LEVELS)
            logger.info("%s | %s | noise SI-SNR dB %s | mean %.2f | hi %.3f", path.name, split, line, scores["noise_si_snr"], scores["noise_si_snr_hi"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
