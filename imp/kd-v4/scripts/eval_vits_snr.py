"""Đánh giá theo mức SNR cho các checkpoint student (best_val.pt) trên validation (n^ từ cache).

    python -m scripts.eval_vits_snr --noise-cache /marimo/cache/nhat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewkd_deit import build_cached_loaders, load_teacher_checkpoint  # noqa: E402
from scripts.run_vits_kd import Student, evaluate  # noqa: E402
from src.train import ReviewKDTrainConfig  # noqa: E402

CONFIG = ROOT / "config" / "reviewkd_deit.yaml"
SNRS = (-5, 0, 5, 10, 15, 20)
# tên run: (kiến trúc timm, số mel, post-LN alpha)
RUNS = {
    "f-r0-preln": ("vit_small_patch16_224", 128, False),
    "f-b-alphapre-full": ("vit_small_patch16_224", 128, True),
    "t-deit-ti-b": ("deit_tiny_patch16_224", 128, True),
    "t-vit-ti-b": ("vit_tiny_patch16_224", 128, True),
    "t-ced-tiny-pre1": ("deit_tiny_patch16_224", 64, False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "output" / "vits-kd")
    parser.add_argument("--checkpoint", default="best_val.pt")
    args = parser.parse_args()
    raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ReviewKDTrainConfig.from_dict({**raw["train"], "eval_batch_size": 64})
    loaders = build_cached_loaders(raw, config, args.noise_cache, None, 4)
    teacher_path = ROOT / raw["teacher"]["checkpoint"]
    front, teacher, _ = load_teacher_checkpoint(teacher_path, device, use_frcrn=False)
    teacher.eval()
    model = raw["model"]
    results = {}
    for name, (arch, n_mels, post_ln) in RUNS.items():
        path = args.runs_dir / name / args.checkpoint
        if not path.exists():
            print(f"SKIP {name}: no {args.checkpoint}", flush=True)
            continue
        specaug = dict(freq_masks=0, freq_width=1, time_masks=0, time_width=1)
        student = Student({**raw["frontend"], "n_mels": n_mels}, specaug, num_classes=36, pretrained=False, freq_bins=n_mels,
                          arch=arch, post_ln=post_ln, dropout=0.0, attn_dropout=0.0, drop_path=0.0)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["state_dict"])
        scores = evaluate(student.to(device), front, loaders["validation"], device)
        results[name] = {
            "epoch": checkpoint.get("epoch"),
            "overall_acc": scores["macro_acc"],
            "overall_mAP": scores["mAP"],
            "acc_by_snr": {s: scores[f"acc_snr_{s}"] for s in SNRS},
            "mAP_by_snr": {s: scores[f"mAP_snr_{s}"] for s in SNRS},
        }
        r = results[name]
        print(f"SNR {name} ep{r['epoch']} overall {r['overall_acc']:.3f}/{r['overall_mAP']:.3f} | acc " +
              " ".join(f"{s}:{r['acc_by_snr'][s]:.3f}" for s in SNRS) + " | mAP " +
              " ".join(f"{s}:{r['mAP_by_snr'][s]:.3f}" for s in SNRS), flush=True)
    out = args.runs_dir / "eval_by_snr.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
