"""
CLI script thực thi và đánh giá hệ thống MoE Champion Pipeline.

Quy trình hoạt động:
    1. Đọc cấu hình mô hình từ config/moe.yaml và config/beat.yaml.
    2. Nạp trọng số 4 động cơ nơ-ron:
       - S1: CED-Tiny Low-Student checkpoint
       - S2: CED-Tiny High-Student checkpoint
       - T1: BEATs Teacher 12L Fold0 checkpoint
       - T2: BEATs Teacher 12L EMA standalone checkpoint
    3. Tải tập kiểm định (validation set) từ cache npy và chạy suy luận hỗn hợp MoE.
    4. Xuất bảng tổng hợp 9 chỉ số chất lượng và ghi file summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ChampionPipeline, evaluate_pipeline, load_pipeline

# Đường dẫn mặc định cấu hình và checkpoints trên môi trường Marimo/Sandbox
DEFAULT_CONFIG = ROOT / "config" / "moe.yaml"
DEFAULT_BEAT_CONFIG = ROOT / "config" / "beat.yaml"
DEFAULT_LOW_CKPT = Path("/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt")
DEFAULT_HIGH_CKPT = Path("/marimo/output/high_expert/last_best_checkpoint.pt")
DEFAULT_TEACHER_FOLD0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
DEFAULT_TEACHER_EMA = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")
DEFAULT_DATA_ROOT = Path("/marimo/dataset/mix-dataset-v2")
DEFAULT_CACHE_DIR = Path("/marimo/dataset/cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MoE Champion Pipeline (Dual-Expert + Dual-Teacher)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Đường dẫn file cấu hình moe.yaml")
    parser.add_argument("--beat-config", type=Path, default=DEFAULT_BEAT_CONFIG, help="Đường dẫn cấu hình beat.yaml cho Teachers")
    parser.add_argument("--low-ckpt", type=Path, default=DEFAULT_LOW_CKPT, help="Checkpoint S1: Low-Expert Student")
    parser.add_argument("--high-ckpt", type=Path, default=DEFAULT_HIGH_CKPT, help="Checkpoint S2: High-Expert Student")
    parser.add_argument("--teacher-fold0", type=Path, default=DEFAULT_TEACHER_FOLD0, help="Checkpoint T1: Teacher Fold0")
    parser.add_argument("--teacher-ema", type=Path, default=DEFAULT_TEACHER_EMA, help="Checkpoint T2: Teacher EMA")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Thư mục gốc chứa dataset âm thanh")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Thư mục cache file .npy")
    parser.add_argument("--split", default="validation", help="Phân vùng dữ liệu cần đánh giá (validation / test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch đánh giá")
    parser.add_argument("--device", default=None, help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    parser.add_argument("--out-dir", type=Path, default=Path("/marimo/output/moe/champion_pipeline"), help="Thư mục lưu kết quả summary.json")
    args = parser.parse_args()

    # Thiết lập thiết bị tính toán tối ưu
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")

    # Khởi tạo pipeline và nạp trọng số
    pipeline = load_pipeline(
        args.config,
        args.low_ckpt,
        args.high_ckpt,
        args.teacher_fold0,
        args.teacher_ema,
        dev,
        beat_config_path=args.beat_config,
    )

    # Đánh giá toàn diện trên tập dữ liệu
    scores = evaluate_pipeline(pipeline, args.data_root, args.cache_dir, args.split, args.batch_size)

    # In bảng tổng hợp kết quả
    print("=============================================================================")
    print("CHAMPION PIPELINE EVALUATION SUMMARY:")
    print(f"Top-1 Accuracy:  {scores['top1_accuracy'] * 100:.2f}%")
    print(f"Top-3 Accuracy:  {scores['top3_accuracy'] * 100:.2f}%")
    print(f"Balanced Acc:    {scores['balanced_accuracy'] * 100:.2f}%")
    print(f"mAP (Macro):     {scores.get('mAP_macro', 0) * 100:.2f}%")
    print(f"mAP (Overall):   {scores['mAP'] * 100:.2f}%")
    print(f"Macro AUC:       {scores['macro_auc'] * 100:.2f}%")
    print("=============================================================================")

    # Lưu kết quả ra file JSON
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved evaluation metrics to: {summary_path}")


if __name__ == "__main__":
    main()
