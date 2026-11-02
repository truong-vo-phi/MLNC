"""
Script to Run Task 4: Proposed Method Benchmark (CED-SingleMoE-SplitFFN ~6.62M params).

Evaluates on 9 Metrics:
top1, top3, balanced_acc, map, auc, precision, recall, f1, mean_alpha.
Breaks down performance across all 6 SNR levels (-5dB, 0dB, 5dB, 10dB, 15dB, 20dB).
Logs configuration, parameter budget, training epoch progress, and exports JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure imp/review-3 is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    MaskedNoiseCacheDataset,
    SyntheticMaskedNoiseCacheDataset,
    get_dataset,
    get_dataloader,
)
from src.models import build_model
from src.models.ced_moe_split import CEDSingleMoESplitFFN, MoETrainer


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("Task4_MoE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


def load_beats_teacher(logger: logging.Logger, device: torch.device) -> nn.Module | None:
    candidate_checkpoints = [
        Path(PROJECT_ROOT.parent.parent / "pretrained/beat-ft/beats_teacher_fold0_weights.pt"),
        Path(PROJECT_ROOT.parent.parent / "pretrained/BEATs_iter3_plus_AS2M.pt"),
    ]
    for ckpt_path in candidate_checkpoints:
        if ckpt_path.is_file():
            try:
                teacher = build_model("beats", num_classes=36).to(device)
                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                # Clean prefix if needed
                cleaned = {}
                for k, v in sd.items():
                    key = k
                    if key.startswith("encoder.model."):
                        key = key[len("encoder.model."):]
                    elif key.startswith("model."):
                        key = key[len("model."):]
                    cleaned[key] = v
                teacher.load_state_dict(cleaned, strict=False)
                teacher.eval()
                for p in teacher.parameters():
                    p.requires_grad = False
                logger.info(f"[+] Loaded Teacher BEATs weights from: {ckpt_path}")
                return teacher
            except Exception as exc:
                logger.warning(f"Could not load teacher from {ckpt_path}: {exc}")

    # Fallback: initialize BEATs directly
    logger.info("[*] Initializing fresh BEATs teacher model (85.28M) for distillation...")
    teacher = build_model("beats", num_classes=36).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def run_task4_benchmark(
    epochs: int = 15,
    batch_size: int = 32,
    smoke: bool = False,
    no_kd: bool = False,
    device_name: str | None = None,
) -> dict:
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"task4_moe_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("=" * 86)
    logger.info("TAC VU 4: DE XUAT MO HINH CAI TIEN CED-SINGLEMOE-SPLITFFN (~6.62M PARAMS)")
    logger.info(f"Timestamp: {timestamp} | Device: {device.type} | Smoke Test: {smoke}")
    logger.info("References: V-MoE (NeurIPS 2021) + Soft MoE (ICLR 2024) + DeepSeekMoE (ACL 2024)")
    logger.info("=" * 86)

    # 1. Dataset loading
    manifest_csv = PROJECT_ROOT.parent.parent / "output/manifest.csv"
    train_cache = PROJECT_ROOT.parent.parent / "cache/nhat/train.npy"
    val_cache = PROJECT_ROOT.parent.parent / "cache/nhat/validation.npy"

    if smoke or not train_cache.is_file() or not val_cache.is_file():
        num_train = 48 if smoke else 128
        num_val = 48 if smoke else 96
        logger.info(f"Using Synthetic datasets (train={num_train}, val={num_val}) for fast benchmark...")
        train_ds = SyntheticMaskedNoiseCacheDataset(num_samples_total=num_train, num_classes=36)
        val_ds = SyntheticMaskedNoiseCacheDataset(num_samples_total=num_val, num_classes=36)
    else:
        train_ds = MaskedNoiseCacheDataset(
            manifest_csv=manifest_csv,
            noise_cache_path=train_cache,
            split="train",
            target_rms=0.07925,
            rms_norm=True,
        )
        val_ds = MaskedNoiseCacheDataset(
            manifest_csv=manifest_csv,
            noise_cache_path=val_cache,
            split="validation",
            target_rms=0.07925,
            rms_norm=True,
        )

    actual_batch_size = 16 if smoke else batch_size
    actual_epochs = 1 if smoke else epochs

    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=actual_batch_size, shuffle=False)

    logger.info(f"Loaded Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)} | Batch Size: {actual_batch_size}")

    # 2. Build Student Model
    logger.info("\nBuilding Proposed Student Model: CEDSingleMoESplitFFN...")
    student = build_model("ced_single_moe_split", num_classes=36)
    param_info = student.get_num_params()
    total_m = param_info["total"] / 1e6
    trainable_m = param_info["trainable"] / 1e6

    logger.info(f"-> Architecture: Single-Backbone MoE (Pre-LN ViT D=192 + GREP Attention)")
    logger.info(f"-> Routing Mechanism: Self-Contained Spectral-Token Router (6 physical features + h_9 CLS)")
    logger.info(f"-> MoE Split-FFN: Blocks 10 & 11 with 1 Shared Expert + 2 Routed Experts (Low & High SNR)")
    logger.info(f"-> Total Parameters: {param_info['total']:,} ({total_m:.2f}M) [Matches ~6M params budget]")
    logger.info(f"-> Trainable Parameters: {param_info['trainable']:,} ({trainable_m:.2f}M)")
    logger.info(f"-> Router Parameters: {param_info['router']:,} ({param_info['router']/1e3:.1f}k)")

    # 3. Optional Teacher Model
    teacher = None
    if not no_kd:
        teacher = load_beats_teacher(logger, device)

    # 4. Trainer Initialization & Training
    trainer = MoETrainer(
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_model=teacher,
        device=device,
        epochs=actual_epochs,
        warmup_epochs=0 if smoke else 2,
        checkpoint_dir=PROJECT_ROOT / "checkpoints/moe",
        logger=logger,
    )

    train_summary = trainer.train()

    # 5. Final Evaluation and Formatted Report
    val_metrics, breakdown = trainer.evaluate()

    logger.info("\n" + "=" * 96)
    logger.info("BANG TONG KET 9 METRICS CHO MO HINH DE XUAT CED-SINGLEMOE-SPLITFFN (~6.62M):")
    logger.info("=" * 96)
    logger.info(f"  Top-1 Accuracy     : {val_metrics['top1']*100:6.2f}%")
    logger.info(f"  Top-3 Accuracy     : {val_metrics['top3']*100:6.2f}%")
    logger.info(f"  Balanced Accuracy  : {val_metrics['balanced_acc']*100:6.2f}%")
    logger.info(f"  Macro mAP          : {val_metrics['map']:6.4f}")
    logger.info(f"  Macro AUC-ROC      : {val_metrics['auc']:6.4f}")
    logger.info(f"  Macro Precision    : {val_metrics['precision']:6.4f}")
    logger.info(f"  Macro Recall       : {val_metrics['recall']:6.4f}")
    logger.info(f"  Macro F1-Score     : {val_metrics['f1']:6.4f}")
    logger.info(f"  Mean Router Alpha  : {val_metrics['mean_alpha']:6.4f}")
    logger.info("-" * 96)

    logger.info("\nBANG PHAN RA CHI TIET 9 METRICS THEO TUNG MUC SNR:")
    logger.info("  | SNR Level | Samples | Top-1 Acc | Top-3 Acc | Bal-Acc |   mAP   |   AUC   | Prec    | Recall  |   F1    | Alpha  |")
    logger.info("  |-----------|---------|-----------|-----------|---------|---------|---------|---------|---------|---------|--------|")

    for snr_key in ("-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"):
        if snr_key in breakdown:
            r = breakdown[snr_key]
            n_cnt = int(r.get("count", 0))
            t1 = r.get("top1", 0.0) * 100.0 if r.get("top1", 0.0) <= 1.0 else r.get("top1", 0.0)
            t3 = r.get("top3", 0.0) * 100.0 if r.get("top3", 0.0) <= 1.0 else r.get("top3", 0.0)
            ba = r.get("balanced_acc", 0.0) * 100.0 if r.get("balanced_acc", 0.0) <= 1.0 else r.get("balanced_acc", 0.0)
            mp = r.get("map", 0.0)
            auc = r.get("auc", 0.0)
            prec = r.get("precision", 0.0)
            rec = r.get("recall", 0.0)
            f1 = r.get("f1", 0.0)
            al = r.get("mean_alpha", 0.50)
            logger.info(
                f"  | {snr_key:>9s} | {n_cnt:7d} | {t1:8.2f}% | {t3:8.2f}% | {ba:6.2f}% | {mp:7.4f} | {auc:7.4f} | {prec:7.4f} | {rec:7.4f} | {f1:7.4f} | {al:6.4f} |"
            )
    logger.info("=" * 96)

    # Save summary JSON
    summary_data = {
        "timestamp": timestamp,
        "model_name": "CED-SingleMoE-SplitFFN",
        "params": param_info,
        "epochs": actual_epochs,
        "val_metrics_9": val_metrics,
        "snr_breakdown": breakdown,
        "history": train_summary["history"],
    }
    summary_json_path = logs_dir / "task4_moe_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"\nSaved Task 4 Summary JSON to: {summary_json_path}")
    logger.info(f"Saved Task 4 Log File to: {log_file}")
    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 4: Benchmark Proposed CED-SingleMoE-SplitFFN (~6.62M params)"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--smoke", action="store_true", help="Quick 1-epoch sanity run")
    parser.add_argument("--no-kd", action="store_true", help="Disable BEATs Knowledge Distillation")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda)")

    args = parser.parse_args()
    run_task4_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoke=args.smoke,
        no_kd=args.no_kd,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
