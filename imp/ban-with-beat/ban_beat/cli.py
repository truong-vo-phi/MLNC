from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import PACKAGE_ROOT, REPO_ROOT
from .pipeline import run_pipeline

DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "ban_beat.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BAN-style born-again BEATs students on Mixture-36l")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir / "test" if args.smoke else args.output_dir
    result = run_pipeline(
        args.config,
        output_dir=output_dir,
        device=args.device,
        smoke=args.smoke,
        run_id=args.run_id,
        num_workers=args.num_workers,
    )
    for row in result.table():
        print(row["model"], row.get("top1_accuracy"), row.get("macro_f1_snr"))


if __name__ == "__main__":
    main()
