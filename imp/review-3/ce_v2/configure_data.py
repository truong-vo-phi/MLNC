"""Configure the raw mixture dataset for the v2 CE baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATHS = ROOT / "benchmark_results/team_kd_moe_research_20261102/TEAM_DATA_PATHS.json"


def main() -> None:
    """Validate and record the dataset location before cache preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    labels = (root / "labels.txt").read_text(encoding="utf-8").splitlines()
    if len(labels) != 36 or len(set(labels)) != 36:
        raise ValueError("Expected 36 distinct labels in labels.txt")
    for split, expected in (("train", 30240), ("eval", 3064)):
        manifest = root / split / "manifest.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = sum(1 for _ in csv.DictReader(stream))
        if rows != expected:
            raise ValueError(f"{manifest}: expected {expected} rows, found {rows}")
        if not (root / split / "mixture").is_dir():
            raise FileNotFoundError(f"Missing mixture directory: {root / split / 'mixture'}")
    PATHS.parent.mkdir(parents=True, exist_ok=True)
    PATHS.write_text(json.dumps({"raw_mixture_root": str(root)}, indent=2), encoding="utf-8")
    print(f"Dataset configured: {root}\nPath file: {PATHS}")


if __name__ == "__main__":
    main()
