"""
Model Evaluation Script:
Evaluates one or all trained models across all SNR levels:
- Computes: acc, precision, recall, macro_f1, micro_f1, mAP
- Prints detailed per-SNR breakdown (-5, 0, 5, 10, 15, 20 dB)
- Saves evaluation summary to CSV / JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset
from src.metrics import compute_metrics, compute_snr_breakdown
from src.models import build_model, list_available_models
from src.utils import count_parameters, load_checkpoint, setup_logger


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    data_root: str | None = None,
    noise_cache_path: str | Path | None = None,
    split: str = "validation",
    batch_size: int = 32,
    device: torch.device | None = None,
    num_classes: int = 36,
    apply_rms_norm: bool = True,
    target_rms: float = 0.07925,
    max_samples: int | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Evaluating {model_name}] Loading checkpoint from: {checkpoint_path}")

    model = build_model(model_name, num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dataset = get_dataset(
        root=data_root,
        split=split,
        num_classes=num_classes,
        max_samples=max_samples,
        noise_cache_path=noise_cache_path,
        apply_rms_norm=apply_rms_norm,
        target_rms=target_rms,
    )
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    source_desc = f"FRCRN cache ({noise_cache_path})" if noise_cache_path else f"split '{split}'"
    print(f"Loaded {len(dataset)} samples from {source_desc} (batches: {len(loader)})")

    all_logits, all_targets, all_snrs = [], [], []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            wave = batch["waveform"].to(device)
            target = batch["target"]
            snr = batch["snr_db"].numpy()

            logits = model(wave)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(target.numpy())
            all_snrs.append(snr)
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(loader):
                print(f"  Processed {b_idx + 1}/{len(loader)} batches...", end="\r", flush=True)
    print()

    import numpy as np
    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)

    metrics = compute_metrics(logits_arr, targets_arr, num_classes=num_classes)
    snr_breakdown = compute_snr_breakdown(logits_arr, targets_arr, snrs_arr, num_classes=num_classes)

    total_p, _ = count_parameters(model)
    print(f"Overall Metrics for {model_name} ({total_p/1e6:.2f}M params):")
    print(f"  Acc: {metrics['acc']*100:.2f}% | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f} | Micro-F1: {metrics['micro_f1']:.4f} | mAP: {metrics['mAP']:.4f}")
    print("  SNR Breakdown:")
    for snr_tag, m in snr_breakdown.items():
        print(f"    SNR {snr_tag:>5s}: Acc={m['acc']*100:5.2f}% | F1={m['macro_f1']:.4f} | mAP={m['mAP']:.4f}")

    return {
        "model": model_name,
        "params_m": total_p / 1e6,
        "metrics": metrics,
        "snr_breakdown": snr_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Review-3 Models")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser.add_argument("--no-rms-norm", action="store_true", help="Disable RMS normalization for masked noise")
    parser.add_argument("--target-rms", type=float, default=0.07925, help="Target RMS for normalization")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (validation or test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg_path = PACKAGE_ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg.get("dataset", {}).get("data_root")
    num_classes = cfg.get("dataset", {}).get("num_classes", 36)
    noise_cache = args.noise_cache or cfg.get("noise_cache", {}).get("cache_file") or cfg.get("noise_cache", {}).get("cache_dir")
    apply_rms_norm = not args.no_rms_norm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_eval = list_available_models() if args.model.lower() == "all" else [args.model]
    results = []

    for name in models_to_eval:
        ckpt = PACKAGE_ROOT / "checkpoints" / name / "best_model.pt"
        if ckpt.is_file():
            res = evaluate_model(
                name,
                ckpt,
                data_root=data_root,
                noise_cache_path=noise_cache,
                split=args.split,
                device=device,
                num_classes=num_classes,
                apply_rms_norm=apply_rms_norm,
                target_rms=args.target_rms,
                max_samples=args.max_samples,
            )
            results.append(res)
        else:
            print(f"Skipping {name}: Checkpoint not found at {ckpt}")

    if results:
        out_summary = PACKAGE_ROOT / "logs" / "evaluation_summary.json"
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved evaluation summary for {len(results)} models to {out_summary}")


if __name__ == "__main__":
    main()
