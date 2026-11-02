"""
Evaluation Script for 9 Audio Classifiers on FRCRN-Masked Noise (.npy Cache).
Computes: acc, precision, recall, macro_f1, micro_f1, mAP, and per-SNR breakdown (-5, 0, 5, 10, 15, 20 dB).
Logs full results to timestamped .log files and JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Safe stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import MaskedNoiseCacheDataset, SyntheticMaskedNoiseCacheDataset, get_dataloader
from src.metrics import compute_metrics, compute_snr_breakdown
from src.models import build_model, list_available_models
from src.utils import count_parameters, setup_logger


def evaluate_single_masked_model(
    model_name: str,
    checkpoint_path: Path | str | None,
    noise_cache_path: str | Path | None,
    data_root: str | Path | None = None,
    split: str = "validation",
    batch_size: int = 32,
    device: torch.device | None = None,
    num_classes: int = 36,
    apply_rms_norm: bool = True,
    target_rms: float = 0.07925,
    max_samples: int | None = None,
    allow_untrained: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """
    Evaluates a single model on FRCRN-masked noise input loaded from .npy cache.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Check checkpoint
    ckpt_file: Path | None = None
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.is_file():
            ckpt_file = p
    elif (PACKAGE_ROOT / "checkpoints" / model_name / "best_model.pt").is_file():
        ckpt_file = PACKAGE_ROOT / "checkpoints" / model_name / "best_model.pt"

    if ckpt_file is None and not allow_untrained:
        msg = f"Skipping {model_name}: No checkpoint found at checkpoints/{model_name}/best_model.pt (use --allow-untrained to evaluate with initialized weights)"
        if logger:
            logger.warning(msg)
        else:
            print(f"[!] {msg}")
        return None

    # 2. Build model
    model = build_model(model_name, num_classes=num_classes).to(device)
    if ckpt_file is not None:
        state = torch.load(ckpt_file, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        ckpt_status = f"Loaded from {ckpt_file}"
    else:
        ckpt_status = "Initialized (untrained baseline)"

    model.eval()
    total_p, train_p = count_parameters(model)

    # 3. Load dataset
    if noise_cache_path is not None:
        try:
            dataset = MaskedNoiseCacheDataset(
                npy_path=noise_cache_path,
                manifest_root=data_root,
                split=split,
                num_classes=num_classes,
                apply_rms_norm=apply_rms_norm,
                target_rms=target_rms,
                max_samples=max_samples,
            )
            data_source_desc = f"FRCRN .npy cache: {dataset.npy_file} ({len(dataset)} samples)"
        except Exception as e:
            if logger:
                logger.warning("Could not load real .npy cache (%s), falling back to SyntheticMaskedNoiseCacheDataset", e)
            else:
                print(f"[!] Could not load real .npy cache ({e}), falling back to SyntheticMaskedNoiseCacheDataset")
            count = max_samples if max_samples else 60
            dataset = SyntheticMaskedNoiseCacheDataset(
                num_samples_total=count,
                num_classes=num_classes,
                apply_rms_norm=apply_rms_norm,
                target_rms=target_rms,
            )
            data_source_desc = f"SyntheticMaskedNoiseCacheDataset ({len(dataset)} samples)"
    else:
        count = max_samples if max_samples else 60
        dataset = SyntheticMaskedNoiseCacheDataset(
            num_samples_total=count,
            num_classes=num_classes,
            apply_rms_norm=apply_rms_norm,
            target_rms=target_rms,
        )
        data_source_desc = f"SyntheticMaskedNoiseCacheDataset ({len(dataset)} samples)"

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    if logger:
        logger.info("=" * 86)
        logger.info("EVALUATION ON FRCRN-MASKED NOISE: %s", model_name.upper())
        logger.info("=" * 86)
        logger.info("Checkpoint: %s", ckpt_status)
        logger.info("Parameters: Total=%.2fM, Trainable=%.2fM", total_p / 1e6, train_p / 1e6)
        logger.info("Data Source: %s", data_source_desc)
        logger.info("RMS Normalization: %s (Target RMS = %.5f)", apply_rms_norm, target_rms)
        logger.info("Split: %s | Total Samples: %d | Batch Size: %d | Device: %s",
                    split, len(dataset), batch_size, device)
        logger.info("-" * 86)
    else:
        print(f"\n[Evaluating {model_name} on Masked Noise]")
        print(f"  Checkpoint: {ckpt_status}")
        print(f"  Data Source: {data_source_desc}")
        print(f"  RMS Normalization: {apply_rms_norm} (sigma_0 = {target_rms})")

    # 4. Inference loop
    all_logits, all_targets, all_snrs = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            wave = batch["waveform"].to(device)
            target = batch["target"]
            snr = batch["snr_db"].numpy()

            logits = model(wave)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(target.numpy())
            all_snrs.append(snr)

    eval_time = time.time() - start_time
    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)

    # 5. Compute metrics
    metrics = compute_metrics(logits_arr, targets_arr, num_classes=num_classes)
    snr_breakdown = compute_snr_breakdown(logits_arr, targets_arr, snrs_arr, num_classes=num_classes)

    # 6. Log results
    if logger:
        logger.info("OVERALL METRICS FOR %s (Evaluated in %.2fs):", model_name.upper(), eval_time)
        logger.info("  Top-1 Accuracy : %6.2f%%", metrics["acc"] * 100)
        logger.info("  Precision      : %8.4f", metrics["precision"])
        logger.info("  Recall         : %8.4f", metrics["recall"])
        logger.info("  Macro-F1       : %8.4f", metrics["macro_f1"])
        logger.info("  Micro-F1       : %8.4f", metrics["micro_f1"])
        logger.info("  mAP            : %8.4f", metrics["mAP"])
        logger.info("-" * 86)
        logger.info("DETAILED SNR BREAKDOWN ON MASKED NOISE:")
        logger.info("  | SNR Level | Samples | Accuracy | Precision |  Recall  | Macro-F1 | Micro-F1 |   mAP    |")
        logger.info("  |-----------|---------|----------|-----------|----------|----------|----------|----------|")
        for snr_tag, m in snr_breakdown.items():
            tag = snr_tag if "dB" in snr_tag else f"{snr_tag}dB"
            logger.info(
                "  | %9s | %7d | %7.2f%% | %9.4f | %8.4f | %8.4f | %8.4f | %8.4f |",
                tag,
                m.get("count", 0),
                m["acc"] * 100,
                m["precision"],
                m["recall"],
                m["macro_f1"],
                m["micro_f1"],
                m["mAP"],
            )
        logger.info("=" * 86 + "\n")
    else:
        print(f"Overall Metrics: Acc={metrics['acc']*100:.2f}% | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | Macro-F1={metrics['macro_f1']:.4f} | Micro-F1={metrics['micro_f1']:.4f} | mAP={metrics['mAP']:.4f}")
        print("SNR Breakdown:")
        for snr_tag, m in snr_breakdown.items():
            print(f"  SNR {snr_tag:>5s} dB (n={m['count']:3d}): Acc={m['acc']*100:5.2f}% | F1={m['macro_f1']:.4f} | mAP={m['mAP']:.4f}")

    return {
        "model": model_name,
        "params_m": round(total_p / 1e6, 2),
        "checkpoint": str(ckpt_file) if ckpt_file else None,
        "rms_normalized": apply_rms_norm,
        "target_rms": target_rms,
        "num_samples": len(dataset),
        "eval_time_s": round(eval_time, 2),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "snr_breakdown": snr_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Review-3 Models on FRCRN-Masked Noise (.npy Cache)")
    parser.add_argument("--model", type=str, default="all", help="Model name (e.g. pann_cnn10, ced_tiny) or 'all'")
    parser.add_argument("--noise-cache", type=str, default=None, help="Path to .npy file or folder with <split>.npy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt file)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (validation, test, train)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--no-rms-norm", action="store_true", help="Disable RMS energy normalization")
    parser.add_argument("--target-rms", type=float, default=0.07925, help="Target RMS value (default: 0.07925)")
    parser.add_argument("--allow-untrained", action="store_true", help="Evaluate untrained model baseline if checkpoint missing")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for output logs")
    args = parser.parse_args()

    # Load config
    cfg_path = PACKAGE_ROOT / args.config
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_root = cfg.get("dataset", {}).get("data_root")
    num_classes = cfg.get("dataset", {}).get("num_classes", 36)
    noise_cache = args.noise_cache or cfg.get("noise_cache", {}).get("cache_file") or cfg.get("noise_cache", {}).get("cache_dir")
    apply_rms_norm = not args.no_rms_norm
    if "apply_rms_norm" in cfg.get("noise_cache", {}) and not args.no_rms_norm:
        apply_rms_norm = cfg["noise_cache"]["apply_rms_norm"]

    device_str = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Setup master logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PACKAGE_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_masked_{args.model}_{timestamp}.log"
    logger = setup_logger(f"eval_masked_{timestamp}", log_file=log_file)

    logger.info("=" * 86)
    logger.info("REVIEW-3: EVALUATION PIPELINE ON FRCRN-MASKED NOISE (.npy)")
    logger.info("Timestamp: %s | Host: Windows | Device: %s", timestamp, device)
    logger.info("Noise Cache Path: %s", noise_cache)
    logger.info("Data Root: %s | Split: %s | Target RMS: %.5f", data_root, args.split, args.target_rms)
    logger.info("=" * 86)

    models_to_eval = list_available_models() if args.model.lower() == "all" else [args.model]
    results: list[dict[str, Any]] = []

    for name in models_to_eval:
        ckpt = args.checkpoint if (args.checkpoint and len(models_to_eval) == 1) else None
        res = evaluate_single_masked_model(
            model_name=name,
            checkpoint_path=ckpt,
            noise_cache_path=noise_cache,
            data_root=data_root,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            num_classes=num_classes,
            apply_rms_norm=apply_rms_norm,
            target_rms=args.target_rms,
            max_samples=args.max_samples,
            allow_untrained=args.allow_untrained,
            logger=logger,
        )
        if res is not None:
            results.append(res)

    # Summary table across models if multiple evaluated
    if len(results) > 1:
        logger.info("=" * 86)
        logger.info("SUMMARY COMPARISON OF ALL EVALUATED MODELS ON FRCRN-MASKED NOISE")
        logger.info("=" * 86)
        logger.info("  | Model Name      | Params(M) | Accuracy | Precision |  Recall  | Macro-F1 | Micro-F1 |   mAP    |")
        logger.info("  |-----------------|-----------|----------|-----------|----------|----------|----------|----------|")
        for r in sorted(results, key=lambda x: x["metrics"]["macro_f1"], reverse=True):
            m = r["metrics"]
            logger.info(
                "  | %-15s | %9.2f | %7.2f%% | %9.4f | %8.4f | %8.4f | %8.4f | %8.4f |",
                r["model"],
                r["params_m"],
                m["acc"] * 100,
                m["precision"],
                m["recall"],
                m["macro_f1"],
                m["micro_f1"],
                m["mAP"],
            )
        logger.info("=" * 86)

    # Save JSON summary
    summary_path = log_dir / "eval_masked_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved complete JSON evaluation summary to: %s", summary_path)
    logger.info("Full evaluation log saved to: %s", log_file)
    print(f"\n[OK] Evaluation completed for {len(results)} model(s). Full log: {log_file}")


if __name__ == "__main__":
    main()
