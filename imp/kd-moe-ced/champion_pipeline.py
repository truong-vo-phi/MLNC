"""
Giao diện dòng lệnh (CLI) cho Champion Pipeline của mô hình CEDSingleMoESplitFFN.
Hỗ trợ chế độ chạy độc lập (Standalone Student) hoặc Ensemble động với Dual-Teachers.

Cách dùng lệnh (CLI Commands):
    # 1. Chạy suy luận độc lập mô hình học sinh (Standalone Student, Zero-Oracle, ~6.62M tham số):
    python champion_pipeline.py --mode standalone

    # 2. Chạy suy luận kết hợp Ensemble động với hai giáo viên BEATs:
    python champion_pipeline.py --mode self_contained

    # 3. Tùy chỉnh đường dẫn checkpoint và kích thước batch:
    python champion_pipeline.py --mode standalone --student-ckpt checkpoints/best_model.pt --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.pipeline import ChampionPipeline, load_pipeline

DEFAULT_STUDENT_CKPT = (
    Path("/marimo/checkpoints/best_champion_moe.pt")
    if Path("/marimo/checkpoints/best_champion_moe.pt").is_file()
    else Path("/marimo/moe-v2/checkpoints/ced-singlemoe-splitffn-kd/best_model.pt")
)
DEFAULT_TEACHER_FOLD0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
DEFAULT_TEACHER_EMA = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")
DEFAULT_BEAT_CONFIG = Path(__file__).resolve().parent / "config" / "beat.yaml"
DEFAULT_DATA_ROOT = Path("/marimo/datasets/mix-dataset-v2")
DEFAULT_CACHE_DIR = Path("/marimo/datasets")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá hiệu năng mô hình CEDSingleMoESplitFFN qua Champion Pipeline"
    )
    parser.add_argument("--student-ckpt", type=Path, default=DEFAULT_STUDENT_CKPT, help="Đường dẫn tới checkpoint trọng số của Student CED-SingleMoE")
    parser.add_argument("--teacher-fold0", type=Path, default=DEFAULT_TEACHER_FOLD0, help="Đường dẫn trọng số giáo viên BEATs Fold0 (chuyên gia SNR thấp)")
    parser.add_argument("--teacher-ema", type=Path, default=DEFAULT_TEACHER_EMA, help="Đường dẫn trọng số giáo viên BEATs EMA (chuyên gia SNR cao)")
    parser.add_argument("--beat-config", type=Path, default=DEFAULT_BEAT_CONFIG, help="Đường dẫn file cấu hình kiến trúc BEATs (beat.yaml)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Thư mục gốc chứa tập dữ liệu Mixture-36")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Thư mục chứa cache file validation.npy và logits giáo viên")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước batch khi suy luận (mặc định: 128)")
    parser.add_argument(
        "--mode",
        choices=["self_contained", "standalone"],
        default="self_contained",
        help="Chế độ suy luận: 'standalone' (chỉ dùng Student CED-SingleMoE) hoặc 'self_contained' (Ensemble động với giáo viên)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output", help="Thư mục xuất báo cáo kết quả")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== CLI Đánh giá Champion Pipeline CED-SingleMoE-SplitFFN ===")
    print(f"Thiết bị tính toán: {device}")
    print(f"Chế độ chạy: {args.mode}")

    val_base = Mixture36Dataset(root=args.data_root, split="eval")
    teacher_cache_dir = args.cache_dir / "teacher_cache"
    has_cache = (
        (teacher_cache_dir / "val_t0_logits.npy").is_file()
        and (teacher_cache_dir / "val_tema_logits.npy").is_file()
    )

    npy_val = args.cache_dir / "validation.npy"
    if not npy_val.is_file():
        print(f"Thông báo: Không tìm thấy file cache kiểm định {npy_val}. Đang chạy thử nghiệm tổng hợp (smoke test).")
        return

    val_dataset = KDNoiseCacheDataset(
        val_base,
        npy_val,
        t0_logits_path=teacher_cache_dir / "val_t0_logits.npy" if has_cache else None,
        tema_logits_path=teacher_cache_dir / "val_tema_logits.npy" if has_cache else None,
    )
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Đã nạp thành công tập kiểm định: {len(val_dataset)} đoạn âm thanh")

    pipeline = load_pipeline(
        student_ckpt=args.student_ckpt,
        teacher_fold0_ckpt=args.teacher_fold0 if not has_cache else None,
        teacher_ema_ckpt=args.teacher_ema if not has_cache else None,
        beat_config_path=args.beat_config,
        device=device,
    )

    all_p_student, all_alphas, all_targets, all_snrs = [], [], [], []

    print("Đang chạy suy luận Student với kỹ thuật 3-Way Temporal TTA (+/-50ms)...")
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
        # Nạp logits của giáo viên từ file cache
        p_t0 = np.load(teacher_cache_dir / "val_t0_logits.npy")
        p_tema = np.load(teacher_cache_dir / "val_tema_logits.npy")
        p_final, _ = pipeline.predict_ensemble(
            torch.empty(0),
            cached_t0_logits=p_t0,
            cached_tema_logits=p_tema,
            p_student=p_student,
            alphas=alphas,
        )

    results = compute_snr_breakdown(p_final, targets, snrs)
    print("\n=== Kết Quả Đánh Giá Chi Tiết Theo Từng Dải SNR ===")
    for snr_key, res in results.items():
        if snr_key == "overall":
            print(f"Tổng thể (Overall) -> Top-1: {res['top1']:.4f} | Top-3: {res['top3']:.4f} | mAP: {res['map']:.4f} | AUC: {res.get('auc', 0):.4f}")
        else:
            print(f"  Dải {snr_key:>5s}: Top-1={res['top1']:.4f} | mAP={res['map']:.4f} (số lượng={res['count']})")


if __name__ == "__main__":
    main()
