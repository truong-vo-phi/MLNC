"""Keep only val/test samples whose noise ytid is in AudioSet eval, never in the AudioSet train splits.

Pretrained models (BEATs AS2M, CED, PANNs, EfficientAT) were trained on balanced/unbalanced train segments,
so samples with those ytids may leak. Original manifests are left untouched.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DATASET_ROOT = Path("D:/content/mix-dataset")
METADATA = Path("D:/content/AudioSet/metadata")
SPLITS = ("validation", "test")
EXPECTED = {"validation": (3064, 281, 6480), "test": (3253, 295, 6480)}
MIN_PER_CLASS = 50


def read_ytids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.split(",", 1)[0].strip() for line in f if line.strip() and not line.startswith("#")}


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean_rows(rows: list[dict[str, str]], eval_ytids: set[str], seen_in_train: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row["noise_ytid"] in eval_ytids and row["noise_ytid"] not in seen_in_train]


def write_clean(rows: list[dict[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "noise_ytid"])
        writer.writerows((row["sample_id"], row["noise_ytid"]) for row in rows)


def report(split: str, rows: list[dict[str, str]], kept: list[dict[str, str]]) -> None:
    ytids = {row["noise_ytid"] for row in kept}
    print(f"\n== {split}: {len(kept)}/{len(rows)} samples, {len(ytids)} ytids")
    by_class = Counter(row["label_names"] for row in kept)
    all_classes = sorted({row["label_names"] for row in rows})
    print(f"classes with samples: {len(by_class)}/{len(all_classes)}")
    for name in all_classes:
        count = by_class.get(name, 0)
        flag = "  <-- below %d" % MIN_PER_CLASS if count < MIN_PER_CLASS else ""
        print(f"  {name:28s} {count:5d} / {sum(r['label_names'] == name for r in rows):5d}{flag}")
    by_snr = Counter(float(row["target_snr_db"]) for row in kept)
    totals = Counter(float(row["target_snr_db"]) for row in rows)
    print("by SNR (dB): " + ", ".join(f"{snr:g}: {by_snr[snr]}/{totals[snr]}" for snr in sorted(totals)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    eval_ytids = read_ytids(args.metadata / "eval_segments.csv")
    seen_in_train = read_ytids(args.metadata / "balanced_train_segments.csv") | read_ytids(
        args.metadata / "unbalanced_train_segments.csv"
    )
    for split in args.splits:
        rows = read_rows(args.root / split / "manifest.csv")
        kept = clean_rows(rows, eval_ytids, seen_in_train)
        report(split, rows, kept)
        expected_kept, expected_ytids, expected_total = EXPECTED[split]
        got = (len(kept), len({row["noise_ytid"] for row in kept}), len(rows))
        if got != (expected_kept, expected_ytids, expected_total):
            raise SystemExit(f"{split}: expected {(expected_kept, expected_ytids, expected_total)}, got {got}; nothing written")
        if not args.no_write:
            out = args.root / f"{split}_clean.csv"
            write_clean(kept, out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
