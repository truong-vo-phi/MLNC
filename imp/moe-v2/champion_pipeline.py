"""
CLI script to evaluate the Single-Backbone MoE v2 Champion Pipeline on Molab Sandbox / GPU.

Usage:
    python champion_pipeline.py --mode self_contained
    python champion_pipeline.py --mode standalone
    python champion_pipeline.py --mode oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.pipeline import ChampionPipelineV2, load_pipeline_v2

DEFAULT_STUDENT_CKPT = Path("/marimo/moe-v2/checkpoints/ced-singlemoe-splitffn-kd/best_model.pt")
DEFAULT_TEACHER_FOLD0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
DEFAULT_TEACHER_EMA = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")
DEFAULT_BEAT_CONFIG = ROOT / "config" / "beat.yaml"
DEFAULT_DATA_ROOT = Path("/marimo/datasets/mix-dataset-v2")
DEFAULT_CACHE_DIR = Path("/marimo/datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Single-Backbone MoE v2 Champion Pipeline")
    parser.add_argument("--student-type", choices=["split_ffn", "hydralora"], default="split_ffn")
    parser.add_argument("--student-ckpt", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher-fold0", type=Path, default=DEFAULT_TEACHER_FOLD0)
    parser.add_argument("--teacher-ema", type=Path, default=DEFAULT_TEACHER_EMA)
    parser.add_argument("--beat-config", type=Path, default=DEFAULT_BEAT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mode", choices=["self_contained", "oracle", "standalone"], default="self_contained")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Single-Backbone MoE v2 Pipeline CLI ===")
    print(f"Device: {device}")
    print(f"Evaluation Mode: {args.mode}")

    # If teacher cache exists, load precomputed teacher logits for fast evaluation
    val_base = Mixture36Dataset(root=args.data_root, split="eval")
    teacher_cache_dir = args.cache_dir / "teacher_cache"
    has_cache = (teacher_cache_dir / "val_t0_logits.npy").is_file() and (teacher_cache_dir / "val_tema_logits.npy").is_file()

    val_dataset = KDNoiseCacheDataset(
        val_base,
        args.cache_dir / "validation.npy",
        t0_logits_path=teacher_cache_dir / "val_t0_logits.npy" if has_cache else None,
        tema_logits_path=teacher_cache_dir / "val_tema_logits.npy" if has_cache else None,
    )
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded validation dataset: {len(val_dataset)} clips")

    # Load pipeline
    pipeline = load_pipeline_v2(
        student_type=args.student_type,
        student_ckpt=args.student_ckpt,
        teacher_fold0_ckpt=args.teacher_fold0 if not has_cache else None,
        teacher_ema_ckpt=args.teacher_ema if not has_cache else None,
        beat_config_path=args.beat_config,
        device=device,
    )

    all_p_student = []
    all_alphas = []
    all_targets = []
    all_snrs = []

    print("Running Student 3-Way Temporal TTA inference...")
    for batch in val_loader:
        w = batch["waveform"].to(device)
        p_s, a = pipeline.predict_student_tta(w)
        all_p_student.append(p_s)
        all_alphas.append(a)
        all_targets.append(batch["target_idx"].numpy())
        all_snrs.append(batch["snr_db"].numpy())

    p_student = np.concatenate(all_p_student, axis=0)
    alphas = np.concatenate(all_alphas, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    snrs = np.concatenate(all_snrs, axis=0)

    if args.mode == "standalone":
        p_final = p_student
    else:
        if has_cache:
            val_t0_log = np.load(teacher_cache_dir / "val_t0_logits.npy")
            val_tema_log = np.load(teacher_cache_dir / "val_tema_logits.npy")
            p_t0 = torch.softmax(torch.from_numpy(val_t0_log) / pipeline.teacher_temp, dim=-1).numpy()
            p_tema = torch.softmax(torch.from_numpy(val_tema_log) / pipeline.teacher_temp, dim=-1).numpy()
        else:
            raise RuntimeError("Teacher cache not found. Please run extract_teacher_logits.py first.")

        if args.mode == "oracle":
            is_high_snr = snrs >= 18.0
            p_final = np.zeros_like(p_student)
            p_final[~is_high_snr] = 0.35 * p_student[~is_high_snr] + 0.45 * p_t0[~is_high_snr] + 0.20 * p_tema[~is_high_snr]
            p_final[is_high_snr] = 0.35 * p_student[is_high_snr] + 0.40 * p_tema[is_high_snr] + 0.25 * p_t0[is_high_snr]
        else:
            # self-contained dynamic soft blending via internal router
            a = alphas[:, None]
            w_t0 = (1.0 - a) * pipeline.t0_low + a * pipeline.t0_high
            w_em = (1.0 - a) * pipeline.tema_low + a * pipeline.tema_high
            p_final = pipeline.student_weight * p_student + w_t0 * p_t0 + w_em * p_tema

    metrics = compute_snr_breakdown(p_final, targets, snrs, num_classes=36)
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS (Mode: {args.mode}, Student: {args.student_type}):")
    print(f"  Top-1 Accuracy:  {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"  Top-3 Accuracy:  {metrics['top3_accuracy'] * 100:.2f}%")
    print(f"  Balanced Acc:    {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Macro mAP:       {metrics['mAP_macro'] * 100:.2f}%")
    print(f"  Macro AUC:       {metrics['macro_auc'] * 100:.2f}%")
    print("-" * 80)
    for snr in [-5, 0, 5, 10, 15, 20]:
        print(f"  SNR {snr:3d} dB: Acc = {metrics.get(f'acc_snr_{snr}', 0) * 100:.2f}%, mAP = {metrics.get(f'mAP_snr_{snr}', 0) * 100:.2f}%")
    print("=" * 80)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"summary_{args.mode}_{args.student_type}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {summary_path}")


if __name__ == "__main__":
    main()
