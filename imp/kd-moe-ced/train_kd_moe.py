"""
Script huấn luyện cho CEDSingleMoESplitFFN với Chưng cất Tri thức Kép (Dual-Teacher Knowledge Distillation).

Cách dùng lệnh (CLI Commands):
    # Huấn luyện mặc định 15 epoch với batch size 64
    python train_kd_moe.py --epochs 15 --batch-size 64

    # Huấn luyện tùy chỉnh tốc độ học và đường dẫn dữ liệu
    python train_kd_moe.py --epochs 20 --batch-size 32 --lr-backbone 3e-5 --lr-experts 1e-4 --lr-router 2e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.kd_trainer import KDTrainer
from src.models.split_moe import CEDSingleMoESplitFFN


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình CEDSingleMoESplitFFN với phương pháp chưng cất tri thức kép (Dual-Teacher KD)"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Tổng số epoch huấn luyện (mặc định: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch huấn luyện (mặc định: 64)")
    parser.add_argument("--lr-backbone", type=float, default=4.0e-5, help="Tốc độ học cho backbone sớm Blocks 0-9 (mặc định: 4e-5)")
    parser.add_argument("--lr-experts", type=float, default=1.2e-4, help="Tốc độ học cho các chuyên gia MoE Blocks 10-11 (mặc định: 1.2e-4)")
    parser.add_argument("--lr-router", type=float, default=2.0e-4, help="Tốc độ học cho bộ định tuyến Router (mặc định: 2e-4)")
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/datasets/mix-dataset-v2"), help="Thư mục gốc chứa tập dữ liệu Mixture-36")
    parser.add_argument("--cache-dir", type=Path, default=Path("/marimo/datasets"), help="Thư mục chứa cache file npy và logits giáo viên")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints", help="Thư mục lưu checkpoint mô hình")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Bắt đầu Huấn luyện CEDSingleMoESplitFFN trên thiết bị: {device} ===")

    train_base = Mixture36Dataset(root=args.data_root, split="train")
    val_base = Mixture36Dataset(root=args.data_root, split="eval")

    teacher_dir = args.cache_dir / "teacher_cache"
    train_npy = args.cache_dir / "train.npy"
    val_npy = args.cache_dir / "validation.npy"

    if not train_npy.is_file():
        print(f"Thông báo: Không tìm thấy file cache dữ liệu tại {train_npy} trên máy cục bộ.")
        print("Mô hình đã được cấu hình hoàn chỉnh và sẵn sàng chạy trên môi trường máy chủ GPU.")
        return

    train_dataset = KDNoiseCacheDataset(
        train_base,
        train_npy,
        t0_logits_path=teacher_dir / "train_t0_logits.npy",
        tema_logits_path=teacher_dir / "train_tema_logits.npy",
    )
    val_dataset = KDNoiseCacheDataset(
        val_base,
        val_npy,
        t0_logits_path=teacher_dir / "val_t0_logits.npy",
        tema_logits_path=teacher_dir / "val_tema_logits.npy",
    )

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=True)

    trainer = KDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_experts=args.lr_experts,
        lr_router=args.lr_router,
        out_dir=args.out_dir,
    )

    for epoch in range(args.epochs):
        train_res = trainer.train_epoch(epoch)
        val_res = trainer.evaluate()
        overall = val_res.get("overall", {})
        top1 = overall.get("top1", 0.0)
        mAP = overall.get("map", 0.0)
        print(f"Epoch {epoch+1:02d} | Val Top-1: {top1:.4f} | Val mAP: {mAP:.4f}")
        is_best = top1 > trainer.best_top1_acc
        if is_best:
            trainer.best_top1_acc = top1
            trainer.best_results = overall
        trainer.save_checkpoint(args.out_dir / f"model_epoch_{epoch+1}.pt", is_best=is_best)


if __name__ == "__main__":
    main()
