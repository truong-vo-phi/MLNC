"""
Comprehensive Evaluation of Standalone CED-HydraLoRA-v3 across 6 SNR levels.
No external teachers (BEATs) are loaded at inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.models.moe_hydra import CEDHydraLoRAV3


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_standalone(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    use_tta: bool = False,
    tta_temp: float = 1.35,
) -> dict[str, dict[str, float]]:
    model.eval()
    all_probs, all_targets, all_snrs, all_alphas = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            wave = batch["waveform"].to(device, non_blocking=True)
            target = batch["target_idx"].numpy()
            snr_db = batch["snr_db"].numpy()

            if use_tta:
                def forward_shifted(shift: int):
                    if shift != 0:
                        if shift > 0:
                            w = F.pad(wave[:, shift:], (0, shift), mode="reflect")
                        else:
                            s = abs(shift)
                            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
                    else:
                        w = wave
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                        out = model(w, return_dict=True)
                    return out["logits"].float(), out["alpha"].float()

                l0, a0 = forward_shifted(0)
                lp, _ = forward_shifted(800)
                ln, _ = forward_shifted(-800)
                l_fused = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
                probs = F.softmax(l_fused / tta_temp, dim=-1).cpu().numpy()
                alphas = a0.cpu().numpy()
            else:
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    out = model(wave, return_dict=True)
                    probs = F.softmax(out["logits"].float(), dim=-1).cpu().numpy()
                    alphas = out["alpha"].float().cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target)
            all_snrs.append(snr_db)
            all_alphas.append(alphas)

    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)
    alphas_arr = np.concatenate(all_alphas, axis=0)

    return compute_snr_breakdown(probs_arr, targets_arr, snrs_arr, alphas=alphas_arr)


def record_results_to_csv(csv_path: Path, row: dict[str, Any]) -> None:
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Standalone CED-HydraLoRA-v3")
    parser.add_argument("--checkpoint", type=Path, default=CURRENT_DIR / "checkpoints" / "best_model.pt")
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/datasets/mix-dataset-v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/marimo/datasets"))
    parser.add_argument("--csv-out", type=Path, default=CURRENT_DIR / "result.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating on {device} ===")

    val_base = Mixture36Dataset(root=args.data_root, split="eval")
    val_dataset = KDNoiseCacheDataset(val_base, args.cache_dir / "validation.npy")
    val_loader = build_dataloader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    ckpt_path = args.checkpoint
    if not ckpt_path.is_file():
        alt_path = CURRENT_DIR / "checkpoints" / "best_adapter_weights.pt"
        if alt_path.is_file():
            ckpt_path = alt_path

    progressive_ranks = True
    ckpt = None
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("adapter_state_dict", ckpt.get("model_state_dict", ckpt))
        # Auto-detect progressive rank vs uniform rank
        sample_key = "vit_base.vit.blocks.4.mlp.fc1.lora_A"
        if sample_key in sd:
            if sd[sample_key].shape[0] == 24:
                progressive_ranks = False
            elif sd[sample_key].shape[0] == 16:
                progressive_ranks = True

    model = CEDHydraLoRAV3(progressive_ranks=progressive_ranks).to(device)
    total_p = count_parameters(model)
    train_p = count_trainable_parameters(model)

    if ckpt is not None:
        sd = ckpt.get("adapter_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint from {ckpt_path} (progressive_ranks={progressive_ranks})")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found! Evaluating raw initialized model.")

    modes = [
        ("CED-HydraLoRA-v3 (Ours: Standalone 1-Pass)", False, "Asymmetric HydraLoRA (r_low=8, r_high=24) + Reliability-Gated Dual-Teacher KD; Standalone 1-Pass (0 BEATs)"),
        ("CED-HydraLoRA-v3 (Ours: Standalone 3-Way TTA + Calib)", True, "Asymmetric HydraLoRA (r_low=8, r_high=24) + Reliability-Gated Dual-Teacher KD + 3-Way Temporal TTA (+/-50ms) + Temp Calibration (T=1.35)"),
    ]

    for name, use_tta, desc in modes:
        print(f"\n==========================================")
        print(f"Running: {name}")
        print(f"==========================================")
        res = evaluate_standalone(model, val_loader, device, use_tta=use_tta)
        overall = res["all"]

        print(f"Overall Top-1:     {overall['top1']*100:.2f}%")
        print(f"Overall Top-3:     {overall['top3']*100:.2f}%")
        print(f"Overall Balanced:  {overall['balanced_acc']*100:.2f}%")
        print(f"Overall Macro mAP: {overall['map']*100:.2f}%")
        print(f"Overall Macro AUC: {overall['auc']*100:.2f}%")
        print(f"Mean Router Alpha: {overall['mean_alpha']:.4f}")

        snr_accs = {}
        for snr in [-5, 0, 5, 10, 15, 20]:
            k = f"{snr}dB"
            if k in res:
                acc = res[k]["top1"] * 100
                m_ap = res[k]["map"] * 100
                a_bar = res[k]["mean_alpha"]
                snr_accs[snr] = acc
                print(f"  SNR {snr:3d} dB: Acc={acc:.2f}%, mAP={m_ap:.2f}%, Mean-Alpha={a_bar:.4f}")

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
            "all_acc": f"{overall['top1']*100:.2f}%",
            "all_top3_acc": f"{overall['top3']*100:.2f}%",
            "all_balanced_acc": f"{overall['balanced_acc']*100:.2f}%",
            "all_mAP": f"{overall['map']*100:.2f}%",
            "all_auc": f"{overall['auc']*100:.2f}%",
        }
        record_results_to_csv(args.csv_out, row)
        print(f"Recorded to {args.csv_out}")


if __name__ == "__main__":
    main()
