import sys
sys.path.insert(0, "/marimo/moe-v2")

from pathlib import Path
import csv
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.models import CEDHydraLoRAMoE, CEDSingleMoESplitFFN, CEDTinyBaseline, DeiTTinyBaseline, ViTSSmallBaseline
from run_experiments import evaluate_model, count_parameters, count_trainable_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running TTA evaluation on {device}...")

val_base = Mixture36Dataset(root=Path("/marimo/datasets/mix-dataset-v2"), split="eval")
val_cache = NoiseCacheDataset(val_base, Path("/marimo/datasets/validation.npy"))
val_loader = build_dataloader(val_cache, batch_size=64, shuffle=False, num_workers=0)

models_to_eval = [
    (
        "CED-HydraLoRA-MoE (3-Way TTA + Calibration)",
        "HydraLoRA (NeurIPS 2024 Oral); Shared A + Asymmetric B_low/B_high (r=16) on Blocks 6-11 + 3-Way TTA (+/-50ms) + Temp Calibration (T=1.35)",
        CEDHydraLoRAMoE(num_classes=36),
        "/marimo/moe-v2/checkpoints/ced-hydralora-moe/best_model.pt",
        True,
    ),
    (
        "CED-SingleMoE-SplitFFN (3-Way TTA + Calibration)",
        "DeepSeekMoE (ACL 2024) + Soft MoE (ICLR 2024) + V-MoE (NeurIPS 2021); Last-2 Blocks Split-FFN + 3-Way TTA (+/-50ms) + Temp Calibration (T=1.35)",
        CEDSingleMoESplitFFN(num_classes=36),
        "/marimo/moe-v2/checkpoints/ced-singlemoe-splitffn/best_model.pt",
        True,
    ),
    (
        "CED-Tiny Baseline (3-Way TTA + Calibration)",
        "Dinkel et al. ICASSP 2024; AudioSet Pretrained + 2D Relative Bias + 3-Way TTA (+/-50ms) + Temp Calibration (T=1.35)",
        CEDTinyBaseline(num_classes=36),
        "/marimo/moe-v2/checkpoints/ced-tiny/best_model.pt",
        True,
    ),
]

csv_path = Path("/marimo/moe-v2/result.csv")
for name, config_desc, model, ckpt_p, use_tta in models_to_eval:
    print(f"\nEvaluating {name}...")
    ckpt = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    
    total_p = count_parameters(model)
    train_p = count_trainable_parameters(model)
    
    scores = evaluate_model(model, val_loader, device, use_tta=use_tta, tta_temp=1.35)
    print(f"RESULTS FOR {name}:")
    print(f"  Top-1 Accuracy:  {scores.get('top1_accuracy', 0.0) * 100:.2f}%")
    print(f"  Top-3 Accuracy:  {scores.get('top3_accuracy', 0.0) * 100:.2f}%")
    print(f"  Balanced Acc:    {scores.get('balanced_accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro mAP:       {scores.get('mAP_macro', 0.0) * 100:.2f}%")
    print(f"  Macro AUC:       {scores.get('macro_auc', 0.0) * 100:.2f}%")
    for snr in [-5, 0, 5, 10, 15, 20]:
        print(f"    SNR {snr:3d} dB: Acc = {scores.get(f'acc_snr_{snr}', 0.0) * 100:.2f}%, mAP = {scores.get(f'mAP_snr_{snr}', 0.0) * 100:.2f}%")
        
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
    
    # Append to result.csv
    fieldnames = list(row.keys())
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
    print(f"Recorded {name} to {csv_path}")

print("All TTA evaluations completed successfully!")
