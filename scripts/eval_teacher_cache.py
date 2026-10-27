"""Đo teacher (BEATs trong best.pt) trên toàn bộ validation bằng n^ đã cache: trần cho mọi student.

    python -m scripts.eval_teacher_cache --noise-cache /marimo/cache/nhat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewkd_deit import CONFIG, build_cached_loaders, load_teacher_checkpoint  # noqa: E402
from src.train import ReviewKDTrainConfig  # noqa: E402
from src.train.reviewkd_trainer import snr_scores  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ReviewKDTrainConfig.from_dict({**raw["train"], "eval_batch_size": args.batch_size})
    loaders = build_cached_loaders(raw, config, args.noise_cache, None, 4)
    front, beats, _ = load_teacher_checkpoint(ROOT / raw["teacher"]["checkpoint"], device, use_frcrn=False)
    beats.eval()
    probabilities, labels, snrs = [], [], []
    for batch in loaders["validation"]:
        logits = beats(front(batch["waveform"].to(device))).logits
        probabilities.append(logits.softmax(-1).float().cpu())
        labels.append(batch["target"].argmax(dim=1))
        snrs.append(batch["snr_db"])
    scores = snr_scores(torch.cat(probabilities).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), 36)
    keys = ["top1_accuracy", "top3_accuracy", "balanced_accuracy", "macro_f1", "mAP", "macro_auc"]
    print("TEACHER_VAL", {k: round(scores[k], 4) for k in keys})
    print("TEACHER_BY_SNR", {k.split("_snr_")[1]: round(v, 3) for k, v in scores.items() if k.startswith("acc_snr_")})


if __name__ == "__main__":
    main()
