"""
Benchmark and Validation Runner for Single-Backbone MoE v2 and ViT Architectures (0-25M params).

References:
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts", NeurIPS 2021. arXiv:2106.05974
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE", ACL 2024. arXiv:2401.06066
    - HydraLoRA: Tian et al., "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning", NeurIPS 2024. arXiv:2404.19245
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957
    - DeiT: Touvron et al., "Training data-efficient image transformers & distillation through attention", ICML 2021. arXiv:2012.12877
    - BEATs: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023. arXiv:2212.09058
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Add current folder to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

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
    """Return total number of parameters in millions (M)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_parameters(model: torch.nn.Module) -> float:
    """Return trainable number of parameters in millions (M)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_tta: bool = False,
    tta_temp: float = 1.35,
) -> dict[str, float]:
    """
    Evaluate model on dataset and return full SNR breakdown.
    Supports 3-Way Temporal Test-Time Augmentation (TTA, +/-50ms) and Temperature Calibration.
    """
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


def append_result_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "config",
        "params_M",
        "trainable_M",
        "snr_-5dB_acc",
        "snr_0dB_acc",
        "snr_5dB_acc",
        "snr_10dB_acc",
        "snr_15dB_acc",
        "snr_20dB_acc",
        "all_acc",
        "all_top3_acc",
        "all_balanced_acc",
        "all_mAP",
        "all_auc",
    ]
    file_exists = csv_path.is_file()
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoE-v2 and ViT Comparative Experiments")
    moe_dir = Path("/marimo/moe-v2") if Path("/marimo/moe-v2").exists() else CURRENT_DIR
    parser.add_argument("--config", type=Path, default=moe_dir / "config" / "config.yaml")
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/datasets/mix-dataset-v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/marimo/datasets"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=moe_dir / "result.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Single-Backbone MoE v2 Experiment Engine ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Prepare Datasets
    print("Loading datasets from:", args.data_root, "and cache:", args.cache_dir)
    train_base = Mixture36Dataset(root=args.data_root, split="train")
    train_cache = NoiseCacheDataset(train_base, args.cache_dir / "train.npy")
    train_loader = build_dataloader(train_cache, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_base = Mixture36Dataset(root=args.data_root, split="eval")
    val_cache = NoiseCacheDataset(val_base, args.cache_dir / "validation.npy")
    val_loader = build_dataloader(val_cache, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_cache)} | Validation samples: {len(val_cache)}")

    models_to_test = [
        # 1. Standard ViT Baseline (ImageNet DeiT-Tiny, ~5.7M)
        (
            "DeiT-Tiny (ImageNet Init, Generic 1D ViT)",
            "Touvron et al. ICML 2021; Standard 1D ViT, D=192, 12L",
            lambda: DeiTTinyBaseline(num_classes=36),
            3,
            False,
            "deit-tiny",
        ),
        # 2. Medium ViT Baseline (ImageNet DeiT-Small, ~22M)
        (
            "ViT-Small / DeiT-Small (22M params)",
            "Touvron et al. ICML 2021; Generic 1D ViT scaled to 22M params, D=384, 12L",
            lambda: ViTSSmallBaseline(num_classes=36),
            3,
            False,
            "vit-small",
        ),
        # 3. Baseline AudioSet CED-Tiny (Single model, 5.42M)
        (
            "CED-Tiny (Single Backbone Baseline)",
            "Dinkel et al. ICASSP 2024; AudioSet Pretrained + 2D Relative Bias, D=192, 12L",
            lambda: CEDTinyBaseline(num_classes=36),
            10,
            False,
            "ced-tiny",
        ),
        # 4. CED-SingleMoE-SplitFFN (Ours: Last-2 Blocks Split-FFN + Shared Expert, 6.62M)
        (
            "CED-SingleMoE-SplitFFN (Ours: Single Backbone MoE)",
            "V-MoE (NeurIPS 2021) + Soft MoE (ICLR 2024) + DeepSeekMoE (ACL 2024); Normalized Split-FFN + Internal Router",
            lambda: CEDSingleMoESplitFFN(num_classes=36),
            12,
            False,
            "ced-singlemoe-splitffn",
        ),
        # 5. CED-HydraLoRA-MoE (Ours: Asymmetric LoRA Experts, 5.72M)
        (
            "CED-HydraLoRA-MoE (Ours: Asymmetric LoRA MoE)",
            "HydraLoRA (NeurIPS 2024 Oral); Shared A + Asymmetric B_low/B_high (r=16) on Blocks 6-11",
            lambda: CEDHydraLoRAMoE(num_classes=36),
            10,
            False,
            "ced-hydralora-moe",
        ),
        # 6. CED-HydraLoRA-MoE (3-Way TTA + Calibration)
        (
            "CED-HydraLoRA-MoE (3-Way TTA + Calibration)",
            "HydraLoRA (NeurIPS 2024 Oral) + 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35)",
            lambda: CEDHydraLoRAMoE(num_classes=36),
            0,
            True,
            "ced-hydralora-moe",
        ),
        # 7. CED-SingleMoE-SplitFFN (3-Way TTA + Calibration)
        (
            "CED-SingleMoE-SplitFFN (3-Way TTA + Calibration)",
            "DeepSeekMoE (ACL 2024) + Soft MoE (ICLR 2024) + 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35)",
            lambda: CEDSingleMoESplitFFN(num_classes=36),
            0,
            True,
            "ced-singlemoe-splitffn",
        ),
    ]

    existing_names = []
    if args.output_csv.is_file():
        try:
            df_exist = pd.read_csv(args.output_csv)
            existing_names = df_exist["name"].tolist()
        except Exception:
            existing_names = []

    for name, config_desc, model_fn, epochs, use_tta, ckpt_name in models_to_test:
        if name in existing_names:
            print(f"Skipping {name} (already evaluated in {args.output_csv})", flush=True)
            continue

        print("\n" + "=" * 80, flush=True)
        print(f"EXPERIMENT: {name}", flush=True)
        print(f"Config: {config_desc}", flush=True)
        model = model_fn()
        total_p = count_parameters(model)
        train_p = count_trainable_parameters(model)
        print(f"Total Params: {total_p:.2f}M | Trainable: {train_p:.2f}M", flush=True)

        ckpt_dir = moe_dir / "checkpoints" / ckpt_name
        ckpt_path = ckpt_dir / "best_model.pt"

        if epochs > 0:
            print(f"Training {name} for {epochs} epochs...", flush=True)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                lr_backbone=5.0e-5,
                lr_experts=1.5e-4,
                lr_router=2.0e-4,
                mixup_alpha=0.4,
                mixup_prob=0.8,
                label_smoothing=0.05,
                lambda_route=0.20,
                amp_dtype="bfloat16",
                out_dir=ckpt_dir,
            )
            scores = trainer.fit()
        else:
            print(f"Direct Evaluation on Clean Validation Set for {name} (use_tta={use_tta})...", flush=True)
            if ckpt_path.is_file():
                print(f"Loading trained weights from {ckpt_path}...")
                ckpt_data = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt_data["model"])
            scores = evaluate_model(model.to(device), val_loader, device, use_tta=use_tta, tta_temp=1.35)

        # Print Evaluation Summary
        print("-" * 80)
        print(f"RESULTS FOR {name}:")
        print(f"  Top-1 Accuracy:  {scores.get('top1_accuracy', 0.0) * 100:.2f}%")
        print(f"  Top-3 Accuracy:  {scores.get('top3_accuracy', 0.0) * 100:.2f}%")
        print(f"  Balanced Acc:    {scores.get('balanced_accuracy', 0.0) * 100:.2f}%")
        print(f"  Macro mAP:       {scores.get('mAP_macro', 0.0) * 100:.2f}%")
        print(f"  Macro AUC:       {scores.get('macro_auc', 0.0) * 100:.2f}%")
        print(f"  SNR breakdown:")
        for snr in [-5, 0, 5, 10, 15, 20]:
            print(f"    SNR {snr:3d} dB: Acc = {scores.get(f'acc_snr_{snr}', 0.0) * 100:.2f}%, mAP = {scores.get(f'mAP_snr_{snr}', 0.0) * 100:.2f}%")
        print("-" * 80)

        # Record to result.csv
        row = {
            "name": name,
            "config": config_desc,
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
        append_result_csv(args.output_csv, row)
        print(f"Recorded results to {args.output_csv}")


if __name__ == "__main__":
    main()
