"""
Training script for CEDSingleMoESplitFFN with Dual-Teacher Knowledge Distillation.

Usage:
    python train_kd_moe.py --epochs 15 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.kd_trainer import KDTrainer
from src.models.split_moe import CEDSingleMoESplitFFN


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CEDSingleMoESplitFFN with Dual-Teacher KD")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-backbone", type=float, default=4.0e-5)
    parser.add_argument("--lr-experts", type=float, default=1.2e-4)
    parser.add_argument("--lr-router", type=float, default=2.0e-4)
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/datasets/mix-dataset-v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/marimo/datasets"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Training CEDSingleMoESplitFFN on {device} ===")

    train_base = Mixture36Dataset(root=args.data_root, split="train")
    val_base = Mixture36Dataset(root=args.data_root, split="eval")

    teacher_dir = args.cache_dir / "teacher_cache"
    train_npy = args.cache_dir / "train.npy"
    val_npy = args.cache_dir / "validation.npy"

    if not train_npy.is_file():
        print(f"Notice: Dataset cache {train_npy} not found on local machine.")
        print("Model is configured and ready for execution in GPU environment.")
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
