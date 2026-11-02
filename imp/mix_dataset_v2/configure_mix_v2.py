"""Generate machine-local dataset paths without committing data or weights."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BEATS = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT/"datasets/team_mix_dataset_v2/extracted/mix-dataset-v2")
    parser.add_argument("--cache-dir", type=Path, default=ROOT/"datasets/team_mix_dataset_v2/extracted")
    parser.add_argument("--weights-dir", type=Path, default=ROOT/"datasets/team_mix_dataset_v2/downloads")
    parser.add_argument("--beats-checkpoint", type=Path, default=ROOT/"checkpoints/pretrained"/DEFAULT_BEATS)
    parser.add_argument("--check", action="store_true", help="Verify required input files, split counts and cache shapes")
    args = parser.parse_args()
    raw, cache, weights = (p.resolve() for p in (args.data_root, args.cache_dir, args.weights_dir))
    paths = {
        "raw_mixture_root": str(raw), "train_manifest": str(raw/"train/manifest.csv"),
        "eval_manifest": str(raw/"eval/manifest.csv"), "test_manifest": str(raw/"test/manifest.csv"),
        "frcrn_cache_root": str(cache), "train_cache": str(cache/"train.npy"),
        "eval_cache": str(cache/"validation.npy"),
        "teacher_fold0": str(weights/"beats_teacher_fold0_weights.pt"),
        "teacher_ema": str(weights/"beats_teacher_ema_standalone.pt"),
        "frcrn_checkpoint": str(weights/"best_e037_20261020T203958Z.pt"),
        "beats_pretrained": str(args.beats_checkpoint.resolve()),
        "source_snapshot": str(ROOT.parents[1]), "student_checkpoint": None,
        "teacher_logits_cache": None, "test_frcrn_cache": None,
    }
    if args.check:
        import numpy as np
        for key in ("train_manifest", "eval_manifest", "train_cache", "eval_cache", "teacher_fold0",
                    "teacher_ema", "frcrn_checkpoint", "beats_pretrained"):
            if not Path(paths[key]).is_file():
                raise FileNotFoundError(f"Missing {key}: {paths[key]}")
        for split, expected in (("train", 30240), ("eval", 3064)):
            with (raw/split/"manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != expected:
                raise ValueError(f"{split}: expected {expected} rows, found {len(rows)}")
            key = "train_cache" if split == "train" else "eval_cache"
            array = np.load(paths[key], mmap_mode="r")
            if array.shape != (expected, 64000):
                raise ValueError(f"Invalid {key} shape: {array.shape}")
    target = ROOT/"benchmark_results/team_kd_moe_research_20261102/TEAM_DATA_PATHS.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote local paths: {target}")
    print("No test manifest or test audio was read.")


if __name__ == "__main__":
    main()
