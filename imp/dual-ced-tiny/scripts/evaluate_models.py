"""
Evaluation Script for Dual-CED-Tiny Models:
Loads best checkpoint and computes 6 metrics with 6-level SNR breakdown.

Two protocols are reported from the same checkpoint:
  * 1-pass: the full 4.0 s clip (checkpoint-selection protocol).
  * overlap: multi-crop inference (default 3 x 3.2 s, starts 0.0/0.4/0.8 s, averaged logits).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloader, get_dataset, rms_normalize
from src.inference import overlap_logits
from src.metrics import compute_metrics, compute_snr_breakdown
from src.models import build_model
from src.utils import load_checkpoint, setup_logger


def _print_metrics(title: str, metrics: dict[str, float], breakdown: dict[str, Any]) -> None:
    print("-" * 80)
    print(title)
    print(f"Top-1 Accuracy:  {metrics['acc'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro Recall:    {metrics['recall']:.4f}")
    print(f"Macro-F1:        {metrics['macro_f1']:.4f}")
    print(f"Micro-F1:        {metrics['micro_f1']:.4f}")
    print(f"Macro mAP:       {metrics['mAP']:.4f}")

    if breakdown:
        print("SNR BREAKDOWN TABLE:")
        print(f"{'SNR':<8} | {'Samples':<8} | {'Accuracy':<10} | {'Macro-F1':<10} | {'mAP':<10}")
        print("-" * 56)
        for snr_key, stats in breakdown.items():
            print(
                f"{snr_key:<8} | {stats['n']:<8} | {stats['acc']*100:6.2f}%    "
                f"| {stats['macro_f1']:8.4f}   | {stats['mAP']:8.4f}"
            )


def evaluate_model(
    model_name: str = "dual_ced_tiny",
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    split: str = "eval",
    batch_size: int = 32,
    device_str: str | None = None,
    use_synthetic: bool = False,
    use_overlap: bool | None = None,
    num_windows: int | None = None,
    crop_seconds: float | None = None,
    aggregate: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path) if config_path else (ROOT / "config" / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_logs = ROOT / config.get("logging", {}).get("log_dir", "logs")
    logger = setup_logger(name=f"eval_{model_name}", log_dir=output_logs)

    # Locate checkpoint
    if checkpoint_path is None:
        checkpoint_root = ROOT / config.get("logging", {}).get("checkpoint_dir", "checkpoints")
        ckpt_candidate = checkpoint_root / model_name / "best_model.pt"
        if ckpt_candidate.is_file():
            checkpoint_path = ckpt_candidate

    router_cfg = config.get("router", {})
    architecture_cfg = config.get("architecture", {}) or {}
    dataset_cfg = config.get("dataset", {})
    input_rms = float((config.get("kd", {}) or {}).get("input_rms", 0.07925))
    sample_rate = int(dataset_cfg.get("sample_rate", 16000))

    overlap_cfg = (config.get("evaluation", {}) or {}).get("overlap", {}) or {}
    if use_overlap is None:
        use_overlap = bool(overlap_cfg.get("enabled", True))
    num_windows = int(num_windows if num_windows is not None else overlap_cfg.get("num_windows", 3))
    crop_seconds = float(crop_seconds if crop_seconds is not None else overlap_cfg.get("crop_seconds", 3.2))
    aggregate = str(aggregate if aggregate is not None else overlap_cfg.get("aggregate", "logits"))

    model = build_model(
        model_name=model_name,
        num_classes=dataset_cfg.get("num_classes", 36),
        sample_rate=sample_rate,
        n_mels=config.get("frontend", {}).get("n_mels", 64),
        num_frames=config.get("frontend", {}).get("num_frames", 398),
        router_transition_center_db=float(router_cfg.get("transition_center_db", 15.0)),
        router_transition_tau=float(router_cfg.get("transition_tau_db", 2.5)),
        router_token_proj_dim=int(router_cfg.get("token_proj_dim", 16)),
        router_hidden_dim=int(router_cfg.get("hidden_dim", 32)),
        use_grep=bool(architecture_cfg.get("use_grep", True)),
        post_ln=bool(architecture_cfg.get("post_ln", True)),
        pre_ln=bool(architecture_cfg.get("pre_ln", True)),
    ).to(device)

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        logger.info("Loading checkpoint from: %s", checkpoint_path)
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        logger.warning("No checkpoint found; evaluating with initialized model weights.")

    model.eval()

    dataset = get_dataset(config, split=split, use_synthetic=use_synthetic)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_overlap_logits = []
    all_targets = []
    all_snrs = []
    all_alphas = []

    with torch.no_grad():
        for batch in loader:
            wave = rms_normalize(batch["waveform"].to(device), target_rms=input_rms)
            target = batch["target"].to(device)
            snr = batch.get("snr_db")
            if snr is not None:
                snr = snr.to(device)

            target_idx = target.argmax(dim=-1) if target.ndim == 2 else target
            details = model(wave, return_details=True)
            if isinstance(details, dict):
                logits = details["logits"]
                all_alphas.append(details["alpha"].float().cpu())
            else:
                logits = details

            all_logits.append(logits.float().cpu())
            all_targets.append(target_idx.cpu())
            if snr is not None:
                all_snrs.append(snr.cpu())

            if use_overlap:
                ov_logits = overlap_logits(
                    model,
                    wave,
                    snr_db=snr,
                    num_windows=num_windows,
                    crop_seconds=crop_seconds,
                    sample_rate=sample_rate,
                    aggregate=aggregate,
                )
                all_overlap_logits.append(ov_logits.cpu())

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    cat_snrs = torch.cat(all_snrs, dim=0) if all_snrs else None

    metrics = compute_metrics(cat_logits, cat_targets, num_classes=model.num_classes)
    breakdown = {}
    if cat_snrs is not None:
        breakdown = compute_snr_breakdown(cat_logits, cat_targets, cat_snrs, num_classes=model.num_classes)

    overlap_metrics: dict[str, float] | None = None
    overlap_breakdown: dict[str, Any] = {}
    if use_overlap:
        cat_overlap = torch.cat(all_overlap_logits, dim=0)
        overlap_metrics = compute_metrics(cat_overlap, cat_targets, num_classes=model.num_classes)
        if cat_snrs is not None:
            overlap_breakdown = compute_snr_breakdown(
                cat_overlap, cat_targets, cat_snrs, num_classes=model.num_classes
            )

    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {model_name.upper()} (Split: {split}, Samples: {len(dataset)})")
    print(f"Input preprocessing: RMS normalize -> {input_rms} (train-consistent)")
    print("=" * 80)
    _print_metrics("1-PASS (full 4.0 s clip, checkpoint-selection protocol)", metrics, breakdown)
    if overlap_metrics is not None:
        _print_metrics(
            f"OVERLAP ({num_windows} x {crop_seconds:.1f} s, aggregate={aggregate})",
            overlap_metrics,
            overlap_breakdown,
        )
    print("=" * 80)

    result: dict[str, Any] = {
        "model": model_name,
        "split": split,
        "num_samples": len(dataset),
        "preprocessing": {"normalization": "rms", "input_rms": input_rms},
        "overlap": {
            "enabled": bool(use_overlap),
            "num_windows": num_windows,
            "crop_seconds": crop_seconds,
            "aggregate": aggregate,
        },
        "overall": metrics,
        "snr_breakdown": breakdown,
    }
    if all_alphas:
        alpha = torch.cat(all_alphas).squeeze(-1)
        result["router"] = {
            "alpha_high_mean": float(alpha.mean()),
            "alpha_high_std": float(alpha.std(unbiased=False)),
        }
        if cat_snrs is not None:
            route_target = torch.sigmoid(
                (cat_snrs.float() - model.router.transition_center_db) / model.router.transition_tau
            )
            result["router"]["soft_target_mae"] = float((alpha - route_target).abs().mean())
            result["router"]["high_branch_acc_at_0_5"] = float(
                ((alpha >= 0.5) == (cat_snrs >= model.router.transition_center_db)).float().mean()
            )
        print(
            "Router: mean alpha_high={alpha_high_mean:.4f} | std={alpha_high_std:.4f}".format(
                **result["router"]
            )
        )
        if "soft_target_mae" in result["router"]:
            print(
                "Router diagnostics: soft-target MAE={soft_target_mae:.4f} | "
                "high gate acc@0.5={high_branch_acc_at_0_5:.4f}".format(**result["router"])
            )
    if overlap_metrics is not None:
        result["overlap_overall"] = overlap_metrics
        result["overlap_snr_breakdown"] = overlap_breakdown

    out_file = output_logs / f"eval_{model_name}_{split}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dual-CED-Tiny Models")
    parser.add_argument("--model", default="dual_ced_tiny", choices=["dual_ced_tiny", "ced_tiny_low", "ced_tiny_high"])
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--split", default="eval", help="Split to evaluate ('eval' or 'test')")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data evaluation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--overlap", dest="overlap", action="store_true", help="Force multi-crop overlap inference")
    parser.add_argument("--no-overlap", dest="overlap", action="store_false", help="Disable multi-crop overlap inference")
    parser.set_defaults(overlap=None)
    parser.add_argument("--num-windows", type=int, default=None, help="Number of overlapping crops (default from config)")
    parser.add_argument("--crop-seconds", type=float, default=None, help="Crop duration in seconds (default from config)")
    parser.add_argument("--aggregate", default=None, choices=["logits", "probs"], help="Overlap aggregation")
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        batch_size=args.batch_size,
        device_str=args.device,
        use_synthetic=args.synthetic,
        use_overlap=args.overlap,
        num_windows=args.num_windows,
        crop_seconds=args.crop_seconds,
        aggregate=args.aggregate,
    )


if __name__ == "__main__":
    main()
