"""Fine-tune FRCRN thành các chuyên gia theo từng mức SNR, tất cả xuất phát từ cùng checkpoint pretrain.

Mỗi mức SNR một lần train riêng (train chỉ trên mức đó, validation đủ mọi mức), một thư mục Drive riêng
`<drive-dir>/snr<mức>/` (best.pt, resume.pt, result.json). Chọn epoch theo SI-SNR của nhiễu tại chính mức đó.
Chạy lần lượt từ cao xuống thấp; dừng giữa chừng thì chạy lại cùng lệnh để nối tiếp (mức đã xong được bỏ qua).

    python -m scripts.run_frcrn_snr_experts --levels 20 --use-drive --drive-dir FA26_Capstone/Truong/frcrn-snr-experts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY  # noqa: E402
from scripts.load_pretrain import download_checkpoint, resolve_url  # noqa: E402
from scripts.run_frcrn_noise import setup_logging  # noqa: E402
from scripts.sweep_frcrn import ORIGINAL_CHECKPOINT, run_from_spec  # noqa: E402

DEFAULT_DRIVE_DIR = "FA26_Capstone/Truong/frcrn-snr-experts"
DEFAULT_LEVELS = (20.0, 15.0, 10.0, 5.0, 0.0, -5.0)

logger = logging.getLogger("run_frcrn_snr_experts")


def stage_name(level: float) -> str:
    return f"snr{level:g}"


def build_spec(args: argparse.Namespace, level: float) -> dict[str, Any]:
    work = Path(args.work_dir)
    return {
        "settings": {
            "target": "speech",
            "seed": args.seed,
            **({"epochs": args.epochs} if args.epochs else {}),
            **({"patience": args.patience} if args.patience else {}),
        },
        "trial_dir": str(work / "checkpoint"),
        "stage_name": stage_name(level),
        "checkpoint": str(ORIGINAL_CHECKPOINT),
        "high_snr_levels": [level],
        "max_samples": args.max_samples,
        "num_workers": args.workers,
        "smoke": args.smoke,
        "use_drive": args.use_drive,
        "drive_dir": args.drive_dir,
        "keep_history": not args.no_history,
        "log_file": str(work / "train.log"),
        "mixdataset_clean": {"seed": args.seed, "train_snr": [level]},
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS), help="các mức SNR, theo thứ tự train")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 64 mẫu: kiểm đường chạy")
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--drive-dir", default=DEFAULT_DRIVE_DIR)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch trong config/frcrn.yaml")
    parser.add_argument("--patience", type=int, default=None, help="ghi đè patience trong config/frcrn.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "output" / "frcrn-snr-experts")
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.work_dir)
    logger.info("mixture-36l: %s", ensure_dataset(MIXTURE_KEY, args.download_root))
    if not ORIGINAL_CHECKPOINT.is_file():
        download_checkpoint(resolve_url("frcrn-pretrain"), ORIGINAL_CHECKPOINT.parent)

    progress_path = args.work_dir / "progress.json"
    progress: dict[str, Any] = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    for level in args.levels:
        name = stage_name(level)
        spec_path = args.work_dir / f"spec_{name}.json"
        spec_path.write_text(json.dumps(build_spec(args, level), indent=2), encoding="utf-8")
        logger.info("=== %s: train trên SNR %g dB, xuất phát từ %s ===", name, level, ORIGINAL_CHECKPOINT)
        result = run_from_spec(spec_path)
        progress[name] = {"level": level, "best_epoch": result.get("best_epoch"), "best_score": result.get("best_score")}
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        logger.info("xong %s: best_epoch=%s best_score=%s", name, result.get("best_epoch"), result.get("best_score"))
    return progress


if __name__ == "__main__":
    main()
