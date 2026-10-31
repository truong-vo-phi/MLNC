"""
Train and Benchmark Single-Backbone MoE v2 models on NVIDIA RTX PRO 6000.
Records full metrics and SNR breakdown directly to result.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

CURRENT_DIR = Path("/marimo/moe-v2") if Path("/marimo/moe-v2").exists() else Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.models import (
    CEDHydraLoRAMoE,
    CEDSingleMoESplitFFN,
    CEDTinyBaseline,
    DeiTTinyBaseline,
    ViTSSmallBaseline,
)
from src.trainer import Trainer


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_tta: bool = False,
    tta_temp: float = 1.35,
) -> dict[str, float]:
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

    return compute_snr_breakdown(probs_arr, targets_arr, snrs_arr, num_classes=36)


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
    print(f"=== Unified Benchmark & MoE-v2 Training Engine ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_root = Path("/marimo/datasets/mix-dataset-v2")
    cache_dir = Path("/marimo/datasets")
    csv_path = CURRENT_DIR / "result.csv"

    print("Loading datasets...")
    train_base = Mixture36Dataset(root=data_root, split="train")
    train_cache = NoiseCacheDataset(train_base, cache_dir / "train.npy")
    train_loader = build_dataloader(train_cache, batch_size=128, shuffle=True, num_workers=0)

    val_base = Mixture36Dataset(root=data_root, split="eval")
    val_cache = NoiseCacheDataset(val_base, cache_dir / "validation.npy")
    val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=0)

    print(f"Dataset ready. Train: {len(train_cache)} | Val: {len(val_cache)}")

    # Models pipeline
    stages = [
        # Stage 1: CED-Tiny Baseline (AudioSet Pretrained + 2D Relative Bias)
        {
            "name": "CED-Tiny (Single Backbone Baseline)",
            "config": "Dinkel et al. ICASSP 2024; AudioSet Pretrained + 2D Relative Bias, D=192, 12L",
            "model_cls": CEDTinyBaseline,
            "ckpt_dir": CURRENT_DIR / "checkpoints" / "ced-tiny",
            "epochs": 10,
            "lr_backbone": 5e-5,
            "lr_experts": 1e-4,
            "lr_router": 1e-4,
        },
        # Stage 2: CED-SingleMoE-SplitFFN (DeepSeekMoE + Soft MoE + V-MoE)
        {
            "name": "CED-SingleMoE-SplitFFN (Ours: Single Backbone MoE)",
            "config": "V-MoE (NeurIPS 2021) + Soft MoE (ICLR 2024) + DeepSeekMoE (ACL 2024); Normalized Split-FFN + Internal Router",
            "model_cls": CEDSingleMoESplitFFN,
            "ckpt_dir": CURRENT_DIR / "checkpoints" / "ced-singlemoe-splitffn",
            "epochs": 10,
            "lr_backbone": 5e-5,
            "lr_experts": 1.5e-4,
            "lr_router": 2.0e-4,
        },
        # Stage 3: CED-HydraLoRA-MoE (HydraLoRA Asymmetric LoRA Experts)
        {
            "name": "CED-HydraLoRA-MoE (Ours: Asymmetric LoRA MoE)",
            "config": "HydraLoRA (NeurIPS 2024 Oral); Shared A + Asymmetric B_low/B_high (r=16) on Blocks 6-11 + Mid-layer Router",
            "model_cls": CEDHydraLoRAMoE,
            "ckpt_dir": CURRENT_DIR / "checkpoints" / "ced-hydralora-moe",
            "epochs": 10,
            "lr_backbone": 5e-5,
            "lr_experts": 1.5e-4,
            "lr_router": 2.0e-4,
        },
    ]

    trained_models = {}

    for stage in stages:
        name = stage["name"]
        print("\n" + "=" * 80)
        print(f"TRAINING STAGE: {name}")
        model = stage["model_cls"](num_classes=36).to(device)
        total_p = count_parameters(model)
        train_p = count_trainable_parameters(model)
        print(f"Total Params: {total_p:.2f}M | Trainable: {train_p:.2f}M")

        trainer = Trainer(
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
            lambda_route=0.20,
            amp_dtype="bfloat16",
            out_dir=stage["ckpt_dir"],
        )
        scores = trainer.fit()
        trained_models[name] = (model, stage["ckpt_dir"] / "best_model.pt", stage["config"], total_p, train_p)

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

    # Stage 4: Test-Time Augmentation (3-Way TTA + Calibration)
    tta_evals = [
        ("CED-SingleMoE-SplitFFN (3-Way TTA + Calibration)", "CED-SingleMoE-SplitFFN (Ours: Single Backbone MoE)", CEDSingleMoESplitFFN),
        ("CED-HydraLoRA-MoE (3-Way TTA + Calibration)", "CED-HydraLoRA-MoE (Ours: Asymmetric LoRA MoE)", CEDHydraLoRAMoE),
    ]

    for tta_name, base_name, model_cls in tta_evals:
        print("\n" + "=" * 80)
        print(f"EVALUATION (3-Way TTA): {tta_name}")
        model, ckpt_p, config_desc, total_p, train_p = trained_models[base_name]
        if ckpt_p.is_file():
            ckpt_data = torch.load(ckpt_p, map_location=device)
            model.load_state_dict(ckpt_data["model"])
        scores_tta = evaluate_model(model.to(device), val_loader, device, use_tta=True, tta_temp=1.35)

        tta_config = f"{config_desc} + 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35)"
        row = {
            "name": tta_name,
            "config": tta_config,
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
        record_result(csv_path, row)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results recorded in {csv_path}")


if __name__ == "__main__":
    main()
