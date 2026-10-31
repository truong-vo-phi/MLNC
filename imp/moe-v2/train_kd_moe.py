"""
End-to-End Training and Benchmarking of Single-Backbone MoE v2 with BEATs Dual-Teacher KD.
Evaluates standalone student, 3-Way TTA + Calibration, and Champion Dynamic Ensemble.
Records all results to result.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

CURRENT_DIR = Path("/marimo/moe-v2") if Path("/marimo/moe-v2").exists() else Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.kd_trainer import KDTrainer
from src.metrics import compute_snr_breakdown
from src.models import (
    CEDHydraLoRAMoE,
    CEDSingleMoESplitFFN,
)


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_student(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_tta: bool = False,
    tta_temp: float = 1.35,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    all_probs, all_targets, all_snrs = [], [], []

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            wave = batch["waveform"].to(device, non_blocking=True)
            target = batch["target_idx"].numpy()
            snr_db = batch["snr_db"].numpy()

            if use_tta:
                def forward_shifted(shift: int) -> torch.Tensor:
                    if shift != 0:
                        if shift > 0:
                            w = F.pad(wave[:, shift:], (0, shift), mode="reflect")
                        else:
                            s = abs(shift)
                            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
                    else:
                        w = wave
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(w)
                        logits = out["logits"] if isinstance(out, dict) else out
                    return logits.float()

                l0 = forward_shifted(0)
                lp = forward_shifted(800)
                ln = forward_shifted(-800)
                l_fused = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
                probs = F.softmax(l_fused / tta_temp, dim=-1).cpu().numpy()
            else:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(wave)
                    logits = out["logits"] if isinstance(out, dict) else out
                    probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target)
            all_snrs.append(snr_db)

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)

    scores = compute_snr_breakdown(probs_arr, targets_arr, snrs_arr, num_classes=36)
    return probs_arr, scores


def record_result(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name", "config", "params_M", "trainable_M",
        "snr_-5dB_acc", "snr_0dB_acc", "snr_5dB_acc", "snr_10dB_acc", "snr_15dB_acc", "snr_20dB_acc",
        "all_acc", "all_top3_acc", "all_balanced_acc", "all_mAP", "all_auc",
    ]
    file_exists = csv_path.is_file()
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Recorded row to {csv_path}: {row['name']} -> Top-1: {row['all_acc']}, mAP: {row['all_mAP']}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Unified Dual-Teacher KD Training & Benchmark Engine ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_root = Path("/marimo/datasets/mix-dataset-v2")
    cache_dir = Path("/marimo/datasets")
    teacher_cache_dir = Path("/marimo/datasets/teacher_cache")
    csv_path = CURRENT_DIR / "result.csv"

    print("Loading datasets with precomputed 3-Way TTA teacher logits...")
    train_base = Mixture36Dataset(root=data_root, split="train")
    train_cache = KDNoiseCacheDataset(
        train_base,
        cache_dir / "train.npy",
        t0_logits_path=teacher_cache_dir / "train_t0_logits.npy",
        tema_logits_path=teacher_cache_dir / "train_tema_logits.npy",
    )
    train_loader = build_dataloader(train_cache, batch_size=128, shuffle=True, num_workers=0)

    val_base = Mixture36Dataset(root=data_root, split="eval")
    val_cache = KDNoiseCacheDataset(
        val_base,
        cache_dir / "validation.npy",
        t0_logits_path=teacher_cache_dir / "val_t0_logits.npy",
        tema_logits_path=teacher_cache_dir / "val_tema_logits.npy",
    )
    val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=0)

    print(f"Dataset ready. Train: {len(train_cache)} | Val: {len(val_cache)}")

    # Load validation teacher probabilities for hybrid ensemble evaluation
    val_t0_log = np.load(teacher_cache_dir / "val_t0_logits.npy")
    val_tema_log = np.load(teacher_cache_dir / "val_tema_logits.npy")
    val_targets = np.array([val_cache[i]["target_idx"] for i in range(len(val_cache))])
    val_snrs = np.array([val_cache[i]["snr_db"] for i in range(len(val_cache))])

    p_t0_cal = F.softmax(torch.from_numpy(val_t0_log) / 1.20, dim=-1).numpy()
    p_tema_cal = F.softmax(torch.from_numpy(val_tema_log) / 1.20, dim=-1).numpy()

    # Model stages to train
    stages = [
        # Stage 1: CED-HydraLoRA-MoE (Dual-Teacher KD)
        {
            "name": "CED-HydraLoRA-MoE (BEATs Dual-Teacher KD)",
            "config": "HydraLoRA (NeurIPS 2024 Oral) + BEATs Dual-Teacher KD (Fold0 Clean + EMA FRCRN); Shared A + Asymmetric B_low/B_high (r=16) on Blocks 6-11",
            "model_cls": CEDHydraLoRAMoE,
            "ckpt_dir": CURRENT_DIR / "checkpoints" / "ced-hydralora-moe-kd",
            "epochs": 12,
            "lr_backbone": 4e-5,
            "lr_experts": 1.2e-4,
            "lr_router": 2.0e-4,
            "lambda_kd": 0.65,
        },
        # Stage 2: CED-SingleMoE-SplitFFN (Dual-Teacher KD)
        {
            "name": "CED-SingleMoE-SplitFFN (BEATs Dual-Teacher KD)",
            "config": "V-MoE (NeurIPS 2021) + Soft MoE (ICLR 2024) + DeepSeekMoE (ACL 2024) + BEATs Dual-Teacher KD; Normalized Split-FFN + Internal Router",
            "model_cls": CEDSingleMoESplitFFN,
            "ckpt_dir": CURRENT_DIR / "checkpoints" / "ced-singlemoe-splitffn-kd",
            "epochs": 12,
            "lr_backbone": 4e-5,
            "lr_experts": 1.2e-4,
            "lr_router": 2.0e-4,
            "lambda_kd": 0.65,
        },
    ]

    for stage in stages:
        name = stage["name"]
        print("\n" + "=" * 80)
        print(f"TRAINING KD STAGE: {name}")
        model = stage["model_cls"](num_classes=36).to(device)
        total_p = count_parameters(model)
        train_p = count_trainable_parameters(model)
        print(f"Total Params: {total_p:.2f}M | Trainable: {train_p:.2f}M")

        trainer = KDTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=stage["epochs"],
            lr_backbone=stage["lr_backbone"],
            lr_experts=stage["lr_experts"],
            lr_router=stage["lr_router"],
            mixup_alpha=0.4,
            mixup_prob=0.8,
            label_smoothing=0.05,
            lambda_kd=stage["lambda_kd"],
            lambda_route=0.20,
            kd_temperature=2.0,
            amp_dtype="bfloat16",
            out_dir=stage["ckpt_dir"],
        )
        scores = trainer.fit()

        # Load best weights for downstream evaluations
        ckpt_path = stage["ckpt_dir"] / "best_model.pt"
        if ckpt_path.is_file():
            ckpt_data = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt_data["model"])

        # 1. Standalone Single-Model (No TTA)
        row = {
            "name": name,
            "config": stage["config"],
            "params_M": round(total_p, 2),
            "trainable_M": round(train_p, 2),
            "snr_-5dB_acc": f"{scores.get('acc_snr_-5', 0.0) * 100:.2f}%",
            "snr_0dB_acc": f"{scores.get('acc_snr_0', 0.0) * 100:.2f}%",
            "snr_5dB_acc": f"{scores.get('acc_snr_5', 0.0) * 100:.2f}%",
            "snr_10dB_acc": f"{scores.get('acc_snr_10', 0.0) * 100:.2f}%",
            "snr_15dB_acc": f"{scores.get('acc_snr_15', 0.0) * 100:.2f}%",
            "snr_20dB_acc": f"{scores.get('acc_snr_20', 0.0) * 100:.2f}%",
            "all_acc": f"{scores.get('top1_accuracy', 0.0) * 100:.2f}%",
            "all_top3_acc": f"{scores.get('top3_accuracy', 0.0) * 100:.2f}%",
            "all_balanced_acc": f"{scores.get('balanced_accuracy', 0.0) * 100:.2f}%",
            "all_mAP": f"{scores.get('mAP_macro', 0.0) * 100:.2f}%",
            "all_auc": f"{scores.get('macro_auc', 0.0) * 100:.2f}%",
        }
        record_result(csv_path, row)

        # 2. Standalone Single-Model (3-Way TTA + Calibration T=1.35)
        print(f"\nEvaluating {name} with 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35)...")
        probs_tta, scores_tta = evaluate_student(model, val_loader, device, use_tta=True, tta_temp=1.35)
        row_tta = {
            "name": f"{name} (3-Way TTA + Calibration)",
            "config": f"{stage['config']} + 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35)",
            "params_M": round(total_p, 2),
            "trainable_M": round(train_p, 2),
            "snr_-5dB_acc": f"{scores_tta.get('acc_snr_-5', 0.0) * 100:.2f}%",
            "snr_0dB_acc": f"{scores_tta.get('acc_snr_0', 0.0) * 100:.2f}%",
            "snr_5dB_acc": f"{scores_tta.get('acc_snr_5', 0.0) * 100:.2f}%",
            "snr_10dB_acc": f"{scores_tta.get('acc_snr_10', 0.0) * 100:.2f}%",
            "snr_15dB_acc": f"{scores_tta.get('acc_snr_15', 0.0) * 100:.2f}%",
            "snr_20dB_acc": f"{scores_tta.get('acc_snr_20', 0.0) * 100:.2f}%",
            "all_acc": f"{scores_tta.get('top1_accuracy', 0.0) * 100:.2f}%",
            "all_top3_acc": f"{scores_tta.get('top3_accuracy', 0.0) * 100:.2f}%",
            "all_balanced_acc": f"{scores_tta.get('balanced_accuracy', 0.0) * 100:.2f}%",
            "all_mAP": f"{scores_tta.get('mAP_macro', 0.0) * 100:.2f}%",
            "all_auc": f"{scores_tta.get('macro_auc', 0.0) * 100:.2f}%",
        }
        record_result(csv_path, row_tta)

        # 3. Champion Pipeline Blended with Teachers (SNR-Dynamic Routing)
        print(f"\nEvaluating {name} in Champion Dynamic MoE Ensemble...")
        is_high_snr = val_snrs >= 18.0
        p_champ = np.zeros_like(probs_tta)

        # Route 1 (Low SNR < 18 dB): 0.35 Student + 0.45 Fold0 + 0.20 EMA
        p_champ[~is_high_snr] = (
            0.35 * probs_tta[~is_high_snr]
            + 0.45 * p_t0_cal[~is_high_snr]
            + 0.20 * p_tema_cal[~is_high_snr]
        )
        # Route 2 (High SNR >= 18 dB): 0.35 Student + 0.40 EMA + 0.25 Fold0
        p_champ[is_high_snr] = (
            0.35 * probs_tta[is_high_snr]
            + 0.40 * p_tema_cal[is_high_snr]
            + 0.25 * p_t0_cal[is_high_snr]
        )

        scores_champ = compute_snr_breakdown(p_champ, val_targets, val_snrs, num_classes=36)
        row_champ = {
            "name": f"Champion Pipeline v2: {name} + BEATs Teachers",
            "config": f"Single-Backbone {name} (3-Way TTA) + Dual BEATs Teachers (Fold0/EMA) + SNR-Dynamic Routing (tau=18dB)",
            "params_M": round(total_p + 180.68, 2),
            "trainable_M": round(train_p, 2),
            "snr_-5dB_acc": f"{scores_champ.get('acc_snr_-5', 0.0) * 100:.2f}%",
            "snr_0dB_acc": f"{scores_champ.get('acc_snr_0', 0.0) * 100:.2f}%",
            "snr_5dB_acc": f"{scores_champ.get('acc_snr_5', 0.0) * 100:.2f}%",
            "snr_10dB_acc": f"{scores_champ.get('acc_snr_10', 0.0) * 100:.2f}%",
            "snr_15dB_acc": f"{scores_champ.get('acc_snr_15', 0.0) * 100:.2f}%",
            "snr_20dB_acc": f"{scores_champ.get('acc_snr_20', 0.0) * 100:.2f}%",
            "all_acc": f"{scores_champ.get('top1_accuracy', 0.0) * 100:.2f}%",
            "all_top3_acc": f"{scores_champ.get('top3_accuracy', 0.0) * 100:.2f}%",
            "all_balanced_acc": f"{scores_champ.get('balanced_accuracy', 0.0) * 100:.2f}%",
            "all_mAP": f"{scores_champ.get('mAP_macro', 0.0) * 100:.2f}%",
            "all_auc": f"{scores_champ.get('macro_auc', 0.0) * 100:.2f}%",
        }
        record_result(csv_path, row_champ)

    print("\n" + "=" * 80)
    print("ALL DUAL-TEACHER KD EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results recorded in {csv_path}")


if __name__ == "__main__":
    main()
