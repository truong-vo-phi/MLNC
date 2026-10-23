"""Fine-tune FRCRN (target noise): tiếng nói của Mixture-36l trộn động với nhiễu AudioSet 36L (bỏ 4 nhãn); sao lưu best.pt/resume.pt lên Drive.

Chạy nền trên molab, ví dụ (lần đầu nên thêm --smoke):
    python -m scripts.run_frcrn_noise --use-drive --drive-dir FA26_Capstone/Truong/frcrn-noise

Cần: biến môi trường KAGGLE_ACCESS_TOKEN (tải dữ liệu), token Drive đã cache (đăng nhập một lần trong notebook),
và (chỉ với --noise-source recovered) bảng offset `offsets_train.csv`, `offsets_validation.csv`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.drive_sync import DriveFileBackup  # noqa: E402
from scripts.load_dataset import AUDIOSET_KEY, MIXTURE_KEY  # noqa: E402
from scripts.load_pretrain import download_checkpoint, resolve_url  # noqa: E402
from scripts.run_manifest import MANIFEST_NAME, RunManifest, drive_listing  # noqa: E402
from scripts.sweep_frcrn import ORIGINAL_CHECKPOINT, run_from_spec  # noqa: E402

DEFAULT_DRIVE_DIR = "FA26_Capstone/Truong/frcrn-noise"
DEFAULT_MAX_ERROR = 1e-2  # giữ hàng có sai số khôi phục <= 1% năng lượng nhiễu (trần SI-SNR đo được ~20 dB)

logger = logging.getLogger("run_frcrn_noise")


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    work = Path(args.work_dir)
    return {
        "settings": {"target": args.target, "seed": args.seed, **({"epochs": args.epochs} if args.epochs else {})},
        "trial_dir": str(work / "checkpoint"),
        "stage_name": args.stage_name,
        "checkpoint": str(ORIGINAL_CHECKPOINT),
        "max_samples": args.max_samples,
        "num_workers": args.workers,
        "smoke": args.smoke,
        "use_drive": args.use_drive,
        "drive_dir": args.drive_dir,
        "keep_history": not args.no_history,
        "log_file": str(work / "train.log"),
        **(
            {"mixdataset_clean": {"seed": args.seed}}
            if args.noise_source == "mixdataset"
            else {"dynamic_36l": {"seed": args.seed}}
            if args.noise_source == "dynamic36"
            else {
                "audioset_noise": {
                    "offsets_dir": str(args.offsets_dir),
                    "cache_dir": str(args.cache_dir),
                    "max_error": args.max_error,
                }
            }
        ),
    }


def setup_logging(work_dir: Path) -> None:
    """Log ra file; chỉ thêm handler stdout khi chưa có (trong notebook marimo đã có sẵn)."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    log_file = str((work_dir / "train.log").resolve())
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in root.handlers):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(fmt)
        root.addHandler(handler)
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)


def main(argv: Sequence[str] | None = None, on_epoch: Callable[[int, dict[str, float]], None] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 64 mẫu: kiểm đường chạy")
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--drive-dir", default=DEFAULT_DRIVE_DIR)
    parser.add_argument("--no-history", action="store_true", help="chỉ ghi đè best.pt, không giữ bản best_e<epoch>_<UTC>.pt")
    parser.add_argument(
        "--noise-source",
        choices=("dynamic36", "recovered", "mixdataset"),
        default="dynamic36",
        help="dynamic36: tiếng nói của mix-dataset trộn động với nhiễu AudioSet 36L (bỏ 4 nhãn); "
        "recovered: mixture cố định + nhiễu khôi phục theo bảng offset; "
        "mixdataset: mixture và clean của mix-dataset, nhiễu tham chiếu = mixture - clean (không cần AudioSet)",
    )
    parser.add_argument("--target", choices=("noise", "speech"), default="noise", help="đối tượng FRCRN ước lượng; đầu ra dùng vẫn là nhiễu")
    parser.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch trong config/frcrn.yaml (đo thời gian: 1)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-name", default="frcrn")
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--offsets-dir", type=Path, default=ROOT / "output" / "audioset-ref")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "datasets" / "audioset_16k")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "output" / "frcrn-noise")
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.work_dir)
    for name in ("offsets_train.csv", "offsets_validation.csv") if args.noise_source == "recovered" else ():
        if not (args.offsets_dir / name).is_file():
            raise FileNotFoundError(f"thiếu {args.offsets_dir / name}: chạy `python -m scripts.audioset_noise_ref offsets` trước")

    # ensure_dataset đặt luôn biến môi trường gốc dữ liệu cho tiến trình này
    for key in (MIXTURE_KEY,) if args.noise_source == "mixdataset" else (MIXTURE_KEY, AUDIOSET_KEY):
        logger.info("%s: %s", key, ensure_dataset(key, args.download_root))
    if not ORIGINAL_CHECKPOINT.is_file():
        download_checkpoint(resolve_url("frcrn-pretrain"), ORIGINAL_CHECKPOINT.parent)

    spec = build_spec(args)
    spec_path = args.work_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    logger.info("spec: %s", spec_path)

    backup = lister = None
    if args.use_drive:
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, token="cache")
        backup = DriveFileBackup(fs, args.drive_dir, MANIFEST_NAME, logger)
        stage_dir = f"{args.drive_dir.strip('/')}/{args.stage_name}"
        lister = lambda: drive_listing(fs, stage_dir)  # noqa: E731
    command = ["python", "-m", "scripts.run_frcrn_noise", *(argv if argv is not None else sys.argv[1:])]
    manifest = RunManifest.start(
        args.work_dir / MANIFEST_NAME,
        spec=spec,
        drive_dir=args.drive_dir if args.use_drive else None,
        stage_name=args.stage_name,
        command=command,
        backup=backup,
        list_drive=lister,
    )
    logger.info("hồ sơ lần chạy: %s (code %s)", manifest.path, manifest.data["code_hash"])

    def track(epoch: int, scores: dict[str, float]) -> None:
        manifest.add_epoch(epoch, scores)
        if on_epoch is not None:
            on_epoch(epoch, scores)

    try:
        result = run_from_spec(spec_path, track)
    except BaseException:
        manifest.finish(None, status="failed")
        raise
    manifest.finish(result)
    (args.work_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("xong: best_epoch=%s best_score=%s", result.get("best_epoch"), result.get("best_score"))
    return result


if __name__ == "__main__":
    main()
