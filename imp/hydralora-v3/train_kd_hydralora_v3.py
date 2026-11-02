"""
Autonomous Training and Benchmarking Runner for CED-HydraLoRA-v3.

Target:
    - Model parameter budget: ~5.80M params (<= 6.0M params).
    - Standalone 1-pass deployment (0 BEATs models at inference).
    - Target Top-1 Accuracy: >= 62% - 63%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import torch

# Ensure local hydralora-v3 is first in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.kd_trainer import ReliabilityGatedKDTrainer
from src.models.moe_hydra import CEDHydraLoRAV3
from evaluate_standalone import evaluate_standalone, record_results_to_csv, count_parameters, count_trainable_parameters


def main():
    parser = argparse.ArgumentParser(description="Train and Benchmark CED-HydraLoRA-v3")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr-backbone", type=float, default=3.5e-5, help="Backbone learning rate")
    parser.add_argument("--lr-lora", type=float, default=1.5e-4, help="LoRA learning rate")
    parser.add_argument("--lr-router", type=float, default=2.0e-4, help="Router learning rate")
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/datasets/mix-dataset-v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/marimo/datasets"))
    parser.add_argument("--out-dir", type=Path, default=CURRENT_DIR / "checkpoints")
    parser.add_argument("--csv-out", type=Path, default=CURRENT_DIR / "result.csv")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================")
    print(f"Starting CED-HydraLoRA-v3 Autonomous Experiment on {device}")
    print(f"================================================================")

    # 1. Datasets and Loaders
    train_base = Mixture36Dataset(root=args.data_root, split="train")
    val_base = Mixture36Dataset(root=args.data_root, split="eval")

    teacher_dir = args.cache_dir / "teacher_cache"
    train_dataset = KDNoiseCacheDataset(
        train_base,
        args.cache_dir / "train.npy",
        t0_logits_path=teacher_dir / "train_t0_logits.npy",
        tema_logits_path=teacher_dir / "train_tema_logits.npy",
    )
    val_dataset = KDNoiseCacheDataset(
        val_base,
        args.cache_dir / "validation.npy",
        t0_logits_path=teacher_dir / "val_t0_logits.npy",
        tema_logits_path=teacher_dir / "val_tema_logits.npy",
    )

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Instantiate Model
    model = CEDHydraLoRAV3(
        num_classes=36,
        target_blocks=(4, 5, 6, 7, 8, 9, 10, 11),
        progressive_ranks=True,
    ).to(device)

    total_p = count_parameters(model)
    train_p = count_trainable_parameters(model)
    print(f"Model: CEDHydraLoRAV3 | Total Params: {total_p:.3f}M | Trainable: {train_p:.3f}M")
    assert total_p < 6.0, f"Model parameters ({total_p:.2f}M) must be strictly < 6.0M!"

    # 3. Initial Baseline Check before training
    print("\n--- Pre-training Initial Validation Probe ---")
    pre_eval = evaluate_standalone(model, val_loader, device, use_tta=False)
    pre_top1 = pre_eval["all"]["top1"] * 100
    print(f"Pre-training Top-1 Acc: {pre_top1:.2f}% (CED-Tiny base init)")

    # 4. Initialize Trainer
    trainer = ReliabilityGatedKDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_lora=args.lr_lora,
        lr_router=args.lr_router,
        warmup_epochs=2,
        kd_temperature=2.0,
        lambda_hard=0.35,
        lambda_kd=0.55,
        lambda_cons=0.15,
        lambda_router=0.10,
        out_dir=args.out_dir,
    )

    history = []
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(args.epochs):
        ep_t0 = time.time()
        train_metrics = trainer.train_epoch(epoch)
        ep_time = time.time() - ep_t0

        # Fast validation
        val_metrics = trainer.evaluate(use_tta=False)
        overall = val_metrics["all"]
        top1 = overall["top1"] * 100
        top3 = overall["top3"] * 100
        mAP = overall["map"] * 100
        auc = overall["auc"] * 100
        mean_alpha = overall["mean_alpha"]

        is_best = top1 > trainer.best_top1_acc * 100
        if is_best:
            trainer.best_top1_acc = top1 / 100.0
            trainer.best_results = overall
            trainer.save_checkpoint(args.out_dir / "best_model.pt", is_best=True)
            adapter_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k or "router" in k or "head_" in k}
            torch.save({"adapter_state_dict": adapter_sd, "top1": top1, "epoch": epoch + 1}, args.out_dir / "best_adapter_weights.pt")

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "reliable_teacher_pct": train_metrics["reliable_teacher_pct"],
            "val_top1": top1,
            "val_top3": top3,
            "val_mAP": mAP,
            "val_auc": auc,
            "mean_alpha": mean_alpha,
            "is_best": is_best,
            "epoch_sec": ep_time,
        }
        history.append(history_entry)

        star = " ★ BEST" if is_best else ""
        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} [{ep_time:.1f}s] | "
            f"Loss: {train_metrics['loss']:.4f} (KD-filter: {train_metrics['reliable_teacher_pct']:.1f}% trusted) | "
            f"Val Top-1: {top1:.2f}% | Top-3: {top3:.2f}% | mAP: {mAP:.2f}% | Alpha: {mean_alpha:.3f}{star}"
        )

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Top-1: {trainer.best_top1_acc * 100:.2f}%")

    # 6. Save Training History
    with open(args.out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 7. Final Rigorous Multi-Mode Evaluation on Best Checkpoint
    best_ckpt_path = args.out_dir / "best_model.pt"
    if best_ckpt_path.is_file():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded best model state from {best_ckpt_path} for final benchmarking.")

    eval_modes = [
        (
            "CED-HydraLoRA-v3 (Ours: Standalone 1-Pass)",
            False,
            "Progressive Asymmetric HydraLoRA (5.89M) + Reliability-Gated Dual-Teacher KD; Standalone 1-Pass (0 BEATs)",
        ),
        (
            "CED-HydraLoRA-v3 (Ours: Standalone 3-Way TTA + Calib)",
            True,
            "Progressive Asymmetric HydraLoRA (5.89M) + Reliability-Gated Dual-Teacher KD + 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35) (0 BEATs)",
        ),
    ]

    final_results = {}
    for name, use_tta, desc in eval_modes:
        print(f"\n========================================================")
        print(f"BENCHMARK: {name}")
        print(f"========================================================")
        res = evaluate_standalone(model, val_loader, device, use_tta=use_tta)
        overall = res["all"]
        final_results[name] = res

        print(f"  Top-1 Accuracy:  {overall['top1'] * 100:.2f}%")
        print(f"  Top-3 Accuracy:  {overall['top3'] * 100:.2f}%")
        print(f"  Balanced Acc:    {overall['balanced_acc'] * 100:.2f}%")
        print(f"  Macro mAP:       {overall['map'] * 100:.2f}%")
        print(f"  Macro AUC:       {overall['auc'] * 100:.2f}%")
        print(f"  Mean Router ᾱ:   {overall['mean_alpha']:.4f}")

        snr_accs = {}
        for snr in [-5, 0, 5, 10, 15, 20]:
            k = f"{snr}dB"
            if k in res:
                acc = res[k]["top1"] * 100
                m_ap = res[k]["map"] * 100
                a_bar = res[k]["mean_alpha"]
                snr_accs[snr] = acc
                print(f"    SNR {snr:3d} dB: Acc={acc:.2f}%, mAP={m_ap:.2f}%, Mean-Alpha={a_bar:.4f}")

        row = {
            "name": name,
            "config": desc,
            "params_M": f"{total_p:.2f}",
            "trainable_M": f"{train_p:.2f}",
            "snr_-5dB_acc": f"{snr_accs.get(-5, 0.0):.2f}%",
            "snr_0dB_acc": f"{snr_accs.get(0, 0.0):.2f}%",
            "snr_5dB_acc": f"{snr_accs.get(5, 0.0):.2f}%",
            "snr_10dB_acc": f"{snr_accs.get(10, 0.0):.2f}%",
            "snr_15dB_acc": f"{snr_accs.get(15, 0.0):.2f}%",
            "snr_20dB_acc": f"{snr_accs.get(20, 0.0):.2f}%",
            "all_acc": f"{overall['top1'] * 100:.2f}%",
            "all_top3_acc": f"{overall['top3'] * 100:.2f}%",
            "all_balanced_acc": f"{overall['balanced_acc'] * 100:.2f}%",
            "all_mAP": f"{overall['map'] * 100:.2f}%",
            "all_auc": f"{overall['auc'] * 100:.2f}%",
        }
        record_results_to_csv(args.csv_out, row)
        print(f"  -> Recorded to {args.csv_out}")

    # 8. Save summary
    summary_path = args.out_dir / "final_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n========================================================")
    print(f"EXPERIMENT FINISHED SUCCESSFULLY!")
    print(f"Best Top-1: {trainer.best_top1_acc * 100:.2f}%")
    print(f"Summary saved to: {summary_path}")
    print(f"Results recorded to: {args.csv_out}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
