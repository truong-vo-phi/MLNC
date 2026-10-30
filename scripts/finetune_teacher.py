"""Finetune BEATs teacher từ checkpoint khởi tạo (beats_mlkd_frcrn_v1, đầu vào n̂ đã chuẩn hoá RMS) trên cache n̂.

    python -m scripts.finetune_teacher --init pretrained/beat-ft/teacher_init.pt --epochs 10 --patience 3

Đầu vào teacher là n̂ từ cache (không chạy FRCRN lúc huấn luyện), chuẩn hoá RMS bằng input_rms của checkpoint.
Chọn epoch theo macro_f1 trên validation. Đầu ra: output/teacher-finetune/{best.pt, resume.pt, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewkd_deit import CONFIG, SharedFront, build_cached_loaders, load_teacher_checkpoint  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.train import BEATTrainConfig, BEATTrainer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

BEATS_CONFIG = ROOT / "config" / "beat.yaml"
CACHE_DIR = ROOT / "cache" / "nhat"
OUT_DIR = ROOT / "output" / "teacher-finetune"
INPUT_RMS = 0.05


class NoiseTeacherTrainer(BEATTrainer):
    def __init__(self, *args: Any, front: SharedFront, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.front = front

    def prepare(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self.front(batch["waveform"].to(self.device))
        return waveform, batch["target"].argmax(dim=1).to(self.device)


def load_finetuned_teacher(
    path: str | Path, device: torch.device
) -> tuple[SharedFront, BEATClassifier, dict[str, Any]]:
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    meta = load_checkpoint(path, beats, device)
    beats.to(device).freeze()
    return SharedFront(None, INPUT_RMS).to(device), beats, meta


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, default=ROOT / "pretrained" / "beat-ft" / "teacher_init.pt")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("finetune_teacher", args.out_dir / "finetune_teacher.log")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BEATTrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        warmup_ratio=0.1,
        batch_size=args.batch_size,
        eval_batch_size=64,
    )
    with open(CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    loaders = build_cached_loaders(raw, config, args.cache_dir, args.max_samples, args.workers)
    front, beats, meta = load_teacher_checkpoint(args.init, device, use_frcrn=False)
    logger.info("init %s | input_rms %.4f | epoch %s", args.init.name, meta["input_rms"], meta["epoch"])

    resume = args.out_dir / "resume.pt"
    trainer = NoiseTeacherTrainer(
        beats, config, device, logger, "teacher", args.out_dir / "best.pt", None, resume, None, None, front=front
    )
    if resume.is_file():
        trainer.load_resume(resume)
    fit = trainer.fit(loaders["train"], loaders["validation"])
    summary = {
        "init": args.init.name,
        "input_rms": float(meta["input_rms"]),
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        "best_macro_f1_validation": fit.best_score,
        "history": trainer.history,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("best epoch %d, validation macro_f1 %.4f", fit.best_epoch, fit.best_score)
    return summary


if __name__ == "__main__":
    main()
