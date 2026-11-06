import os, sys, json, yaml
from pathlib import Path
import numpy as np
import torch

MOE_PROJECT_ROOT = Path("/marimo/moe")
if str(MOE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_PROJECT_ROOT))

# 1. Print training stage summary
s1_path = Path("/marimo/output/moe-from-scratch/20261105-081718-fp32-math/summary.json")
s2_path = Path("/marimo/output/moe-no-rms/summary.json")

s1 = json.loads(s1_path.read_text(encoding="utf-8")) if s1_path.exists() else {}
s2 = json.loads(s2_path.read_text(encoding="utf-8")) if s2_path.exists() else {}

print("=" * 80)
print("=== SO SANH GIAI DOAN HUAN LUYEN (TRAINING SUMMARY) ===")
print("=" * 80)
if s1 and s2:
    print(f"{'Metric / Stage':<35} | {'Run 1 (With RMS)':<20} | {'Run 2 (No RMS)':<20}")
    print("-" * 80)
    print(f"{'Low-Student Best Macro Acc':<35} | {s1['low']['best_score']*100:.2f}% (ep {s1['low']['best_epoch']}){'':<6} | {s2['low']['best_score']*100:.2f}% (ep {s2['low']['best_epoch']})")
    print(f"{'High-Student Best Top-1 Acc':<35} | {s1['high']['best_score']*100:.2f}% (ep {s1['high']['best_epoch']}){'':<6} | {s2['high']['best_score']*100:.2f}% (ep {s2['high']['best_epoch']})")
print("=" * 80)

# 2. Run Full MoE Evaluation for both
from src.pipeline import load_pipeline, evaluate_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("/datasets/mix-dataset-v2")
cache_dir = Path("/datasets/cache-frcrn")

print("\n--- 1. Evaluating Full MoE Pipeline: Run 1 (With RMS) ---")
pipe1 = load_pipeline(
    config_path=MOE_PROJECT_ROOT / "config/moe.yaml",
    low_ckpt="/marimo/output/moe-from-scratch/20261105-081718-fp32-math/low/last_best_checkpoint.pt",
    high_ckpt="/marimo/output/moe-from-scratch/20261105-081718-fp32-math/high/last_best_checkpoint.pt",
    t_fold0_ckpt="/marimo/pretrained/beats_teacher_fold0_weights.pt",
    t_ema_ckpt="/marimo/pretrained/beats_teacher_ema_standalone.pt",
    device=device,
)
scores_run1 = evaluate_pipeline(
    pipeline=pipe1,
    data_root=data_root,
    cache_dir=cache_dir,
    split="validation",
    batch_size=64,
)
del pipe1
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n--- 2. Evaluating Full MoE Pipeline: Run 2 (No RMS) ---")
pipe2 = load_pipeline(
    config_path=MOE_PROJECT_ROOT / "config/moe.yaml",
    low_ckpt="/marimo/output/moe-no-rms/low/last_best_checkpoint.pt",
    high_ckpt="/marimo/output/moe-no-rms/high/last_best_checkpoint.pt",
    t_fold0_ckpt="/marimo/pretrained/beats_teacher_fold0_weights.pt",
    t_ema_ckpt="/marimo/pretrained/beats_teacher_ema_standalone.pt",
    device=device,
)
scores_run2 = evaluate_pipeline(
    pipeline=pipe2,
    data_root=data_root,
    cache_dir=cache_dir,
    split="validation",
    batch_size=64,
)
del pipe2
if torch.cuda.is_available():
    torch.cuda.empty_cache()

comparison = {
    "run1_with_rms": scores_run1,
    "run2_no_rms": scores_run2,
}

Path("/marimo/output/comparison_rms_vs_no_rms.json").write_text(
    json.dumps(comparison, indent=2), encoding="utf-8"
)

print("\n" + "=" * 90)
print("=== KET QUA NGHIEM THU CHINH THUC TOAN BO MOE PIPELINE (3,064 CLIPS) ===")
print("=" * 90)
print(f"{'Chi so (Metric)':<28} | {'Run 1 (With RMS)':<18} | {'Run 2 (No RMS)':<18} | {'Delta (R1 - R2)':<15}")
print("-" * 90)

core_metrics = [
    ("Top-1 Accuracy", "top1_accuracy"),
    ("Top-3 Accuracy", "top3_accuracy"),
    ("Balanced Accuracy", "balanced_accuracy"),
    ("Macro Precision", "precision_macro"),
    ("Macro Recall", "recall_macro"),
    ("Macro F1-Score", "macro_f1"),
    ("Micro F1-Score", "micro_f1"),
    ("Macro mAP", "mAP"),
    ("Macro ROC-AUC", "macro_auc"),
]

for label, key in core_metrics:
    v1 = scores_run1.get(key, 0.0) * 100
    v2 = scores_run2.get(key, 0.0) * 100
    delta = v1 - v2
    sign = "+" if delta >= 0 else ""
    print(f"{label:<28} | {v1:6.2f}%{'':<11} | {v2:6.2f}%{'':<11} | {sign}{delta:6.2f}%")

print("\n" + "=" * 90)
print("=== CHI TIET PHAN RA THEO TUNG MUC SNR (TOP-1 ACCURACY) ===")
print("=" * 90)
print(f"{'SNR Tier':<28} | {'Run 1 (With RMS)':<18} | {'Run 2 (No RMS)':<18} | {'Delta (R1 - R2)':<15}")
print("-" * 90)

for snr_val in [-5, 0, 5, 10, 15, 20]:
    k = f"acc_snr_{snr_val}"
    v1 = scores_run1.get(k, 0.0) * 100
    v2 = scores_run2.get(k, 0.0) * 100
    delta = v1 - v2
    sign = "+" if delta >= 0 else ""
    print(f"SNR {snr_val:+3d} dB{'':<18} | {v1:6.2f}%{'':<11} | {v2:6.2f}%{'':<11} | {sign}{delta:6.2f}%")

print("=" * 90)
