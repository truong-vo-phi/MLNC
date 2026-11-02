"""
Champion Pipeline CLI for CEDSingleMoESplitFFN with Optional Dual-Teacher Ensemble.

Usage:
    python champion_pipeline.py --mode self_contained
    python champion_pipeline.py --mode standalone
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.pipeline import ChampionPipeline, load_pipeline

DEFAULT_STUDENT_CKPT = Path("/marimo/moe-v2/checkpoints/ced-singlemoe-splitffn-kd/best_model.pt")
DEFAULT_TEACHER_FOLD0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
DEFAULT_TEACHER_EMA = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")
DEFAULT_BEAT_CONFIG = Path(__file__).resolve().parent / "config" / "beat.yaml"
DEFAULT_DATA_ROOT = Path("/marimo/datasets/mix-dataset-v2")
DEFAULT_CACHE_DIR = Path("/marimo/datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CEDSingleMoESplitFFN Champion Pipeline")
    parser.add_argument("--student-ckpt", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher-fold0", type=Path, default=DEFAULT_TEACHER_FOLD0)
    parser.add_argument("--teacher-ema", type=Path, default=DEFAULT_TEACHER_EMA)
    parser.add_argument("--beat-config", type=Path, default=DEFAULT_BEAT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mode", choices=["self_contained", "standalone"], default="self_contained")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--limit", type=int, default=None, help="Optional bounded integration check")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== CED-SingleMoE-SplitFFN Pipeline CLI ===")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    val_base = Mixture36Dataset(root=args.data_root, split="eval")
    teacher_cache_dir = args.cache_dir / "teacher_cache"
    has_cache = (
        (teacher_cache_dir / "val_t0_logits.npy").is_file()
        and (teacher_cache_dir / "val_tema_logits.npy").is_file()
    )

    npy_val = args.cache_dir / "validation.npy"
    if not npy_val.is_file():
        print(f"Validation cache {npy_val} not found. Running synthetic smoke test.")
        return

    val_dataset = KDNoiseCacheDataset(
        val_base,
        npy_val,
        t0_logits_path=teacher_cache_dir / "val_t0_logits.npy" if has_cache else None,
        tema_logits_path=teacher_cache_dir / "val_tema_logits.npy" if has_cache else None,
    )
    if args.limit is not None:
        from torch.utils.data import Subset
        groups = {}
        for i in range(len(val_dataset)):
            item = val_base[i]
            key = (int(item["target_idx"]), int(item["snr_db"]))
            groups.setdefault(key, []).append(i)
        selected = []
        for level in range(max(map(len, groups.values()))):
            selected.extend(v[level] for v in groups.values() if len(v) > level)
            if len(selected) >= args.limit:
                break
        val_dataset = Subset(val_dataset, sorted(selected[:args.limit]))
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded validation dataset: {len(val_dataset)} clips")

    pipeline = load_pipeline(
        student_ckpt=args.student_ckpt,
        teacher_fold0_ckpt=args.teacher_fold0 if args.mode == "self_contained" and not has_cache else None,
        teacher_ema_ckpt=args.teacher_ema if args.mode == "self_contained" and not has_cache else None,
        beat_config_path=args.beat_config,
        device=device,
    )

    all_p_student, all_alphas, all_targets, all_snrs = [], [], [], []
    all_final = []

    print("Running Student 3-Way Temporal TTA inference...")
    for batch in val_loader:
        w = batch["waveform"].to(device)
        if args.mode == "self_contained":
            p_s, a = pipeline.predict_ensemble(w, batch.get("t0_logits"), batch.get("tema_logits"))
        else:
            p_s, a = pipeline.predict_student_tta(w)
        all_final.append(p_s)
        all_p_student.append(p_s)
        all_alphas.append(a)
        all_targets.append(batch["target_idx"].numpy())
        all_snrs.append(batch["snr_db"].numpy())

    p_student = np.concatenate(all_p_student, axis=0)
    alphas = np.concatenate(all_alphas, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    snrs = np.concatenate(all_snrs, axis=0)

    p_final = np.concatenate(all_final, axis=0)

    results = compute_snr_breakdown(p_final, targets, snrs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez(args.out_dir / "predictions.npz", probabilities=p_final, targets=targets, snrs=snrs, alphas=alphas)
    print("\n=== Final Benchmark Results ===")
    for snr_key, res in results.items():
        if snr_key == "overall":
            print(f"Overall Top-1: {res['top1']:.4f} | Top-3: {res['top3']:.4f} | mAP: {res['map']:.4f} | AUC: {res.get('auc', 0):.4f}")
        else:
            print(f"  {snr_key:>5s}: Top-1={res['top1']:.4f} | mAP={res['map']:.4f} (count={res['count']})")


if __name__ == "__main__":
    main()
