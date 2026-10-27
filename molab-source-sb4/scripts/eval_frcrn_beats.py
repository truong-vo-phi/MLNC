"""Đo (chỉ suy luận) chất lượng phân loại nhiễu của BEATs đã fine-tune trên các cách tạo đầu vào khác nhau.

Ba điều kiện, cùng BEATs fine-tune (scripts.run_beats_noise), cùng chuẩn hoá RMS:
  frcrn_beats : mixture -> FRCRN (đóng băng, ra n_hat = mixture - tiếng nói ước lượng) -> BEATs
  mixture_only: mixture -> BEATs (không FRCRN)
  noise_oracle: nhiễu thật `noise_path` -> BEATs (trần; chính là đầu vào teacher)
Báo 9 chỉ số tổng và theo từng mức SNR (-5 ... 20 dB) cho validation và test.

    python -m scripts.eval_frcrn_beats --frcrn pretrained/frcrn_best.pt --beats-finetuned output/beats-noise-ft/checkpoint/beat_noise/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY, Mixture36Dataset, build_dataloader  # noqa: E402
from scripts.run_kd_beat import format_table, per_snr_table, write_per_snr_csv  # noqa: E402
from scripts.run_reviewkd import read_reviewkd_config  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.model.reviewkd import rms_normalize  # noqa: E402
from src.train.reviewkd_trainer import snr_scores  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402

CONDITIONS = ("frcrn_beats", "mixture_only", "noise_oracle")
NUM_CLASSES = 36

logger = logging.getLogger("eval_frcrn_beats")


@torch.no_grad()
def evaluate_conditions(
    loader: Any,
    separator: Callable[[Tensor], Tensor],
    predict_proba: Callable[[Tensor], Tensor],
    target_rms: float,
    device: torch.device,
    conditions: Sequence[str] = CONDITIONS,
    desc: str = "eval",
) -> dict[str, dict[str, float]]:
    """Một lượt duyệt loader, tính `snr_scores` cho từng điều kiện đầu vào."""
    probs: dict[str, list[Tensor]] = {name: [] for name in conditions}
    labels, snrs = [], []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        mixture = batch["waveform"].to(device)
        inputs = {
            "frcrn_beats": lambda: separator(mixture),
            "mixture_only": lambda: mixture,
            "noise_oracle": lambda: batch["noise"].to(device),
        }
        for name in conditions:
            probs[name].append(predict_proba(rms_normalize(inputs[name](), target_rms)).float().cpu())
        labels.append(batch["target"].argmax(dim=1).cpu())
        snrs.append(batch["snr_db"].cpu())
    y, s = torch.cat(labels).numpy(), torch.cat(snrs).numpy()
    return {name: snr_scores(torch.cat(chunks).numpy(), y, s, NUM_CLASSES) for name, chunks in probs.items()}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frcrn", type=Path, default=ROOT / "pretrained" / "frcrn_best.pt")
    parser.add_argument("--beats-finetuned", type=Path, required=True)
    parser.add_argument("--reviewkd-config", type=Path, default=ROOT / "config" / "reviewkd.yaml")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS), help="điều kiện đầu vào cần đo")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "eval-frcrn-beats")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    raw = read_reviewkd_config(args.reviewkd_config)
    target_rms = float(raw["review"]["target_rms"])
    duration = float(raw["dataset"]["duration"])
    logger.info("mixture-36l: %s", ensure_dataset(MIXTURE_KEY, args.download_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    separator = FRCRNSeparator.from_yaml(ROOT / "config" / "frcrn.yaml")
    separator.load_pretrained(args.frcrn)
    separator.to(device).freeze()
    beats = BEATClassifier.from_yaml(ROOT / "config" / "beat.yaml")
    load_checkpoint(args.beats_finetuned, beats)
    beats.to(device).freeze()
    logger.info("FRCRN %s | BEATs fine-tuned %s | target_rms %.5f", args.frcrn, args.beats_finetuned, target_rms)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"frcrn": str(args.frcrn), "beats": str(args.beats_finetuned), "target_rms": target_rms, "results": {}}
    for split in args.splits:
        dataset = Mixture36Dataset(split=split, duration=duration, load_noise=True)
        if args.max_samples:
            dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
        loader = build_dataloader(dataset, args.batch_size, False, args.workers)
        scores = evaluate_conditions(loader, separator, beats.predict_proba, target_rms, device, tuple(args.conditions), desc=f"{split}")
        report["results"][split] = {}
        for name, values in scores.items():
            table = per_snr_table(values)
            write_per_snr_csv(table, args.out_dir / f"{split}_{name}.csv")
            report["results"][split][name] = {"overall": {k: values[k] for k in values if "_snr_" not in k}, "per_snr": table}
            logger.info("%s | %s (9 metrics per SNR)%s%s", split, name, chr(10), format_table(table))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
