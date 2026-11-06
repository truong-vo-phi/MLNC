"""
Benchmark Runner for 9 Acoustic Noise Classification Models in Review-3.
Trains each model sequentially on FRCRN-masked noise cache (/datasets/mix-frcrn),
tracks epoch-by-epoch progress with SNR breakdown,
compiles a unified comparison table across all 9 models and 6 SNR levels,
and saves detailed logs and JSON summaries.
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
from typing import Any

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset
from src.models import build_model
from src.trainer import Trainer
from src.utils import count_parameters, set_seed, setup_logger

CANONICAL_9_MODELS = [
    "pann_cnn10",
    "ced_tiny",
    "ced_mini",
    "ced_small",
    "efficient_mn10",
    "vit_s",
    "mobilevit",
    "mobilenet",
    "beats",
]

MODEL_FAMILIES = {
    "pann_cnn10": "CNN (PANNs)",
    "ced_tiny": "Audio Transformer",
    "ced_mini": "Audio Transformer",
    "ced_small": "Audio Transformer",
    "efficient_mn10": "Efficient MobileNet",
    "vit_s": "Vision Transformer",
    "mobilevit": "Hybrid CNN-ViT",
    "mobilenet": "Lightweight CNN",
    "beats": "Acoustic Transformer (BEATs)",
}

MODEL_CONFIG_MAP = {
    "pann_cnn10": "panns_cnn10",
    "ced_tiny": "ced_meanpool",
    "ced_mini": "ced_meanpool",
    "ced_small": "ced_meanpool",
    "efficient_mn10": "efficientat_mn10",
    "vit_s": "hybrid_kdmoe_s",
    "mobilevit": "hybrid_kdmoe_s",
    "mobilenet": "hybrid_kdmoe_s",
    "beats": "teacher",
}


def get_model_profile(model_name: str, models_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extracts model-specific hyperparameter profile from models.yaml."""
    models_dict = models_cfg.get("models", {})
    key = MODEL_CONFIG_MAP.get(model_name, model_name)
    if key == "teacher":
        return {
            "lr": 3.0e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "frontend": {"n_mels": 128, "num_frames": 398},
        }
    profile = dict(models_dict.get(key, {}))
    return profile


def get_models_architecture_table(models: list[str] | None = None) -> list[dict[str, Any]]:
    """Generates parameter architecture summary for all requested models."""
    models = models or CANONICAL_9_MODELS
    table = []
    for idx, name in enumerate(models, 1):
        try:
            m = build_model(name, num_classes=36)
            total_p, train_p = count_parameters(m)
            params_m = round(total_p / 1e6, 2)
            trainable_m = round(train_p / 1e6, 2)
        except Exception:
            params_m = 0.0
            trainable_m = 0.0

        table.append({
            "idx": idx,
            "model_key": name,
            "family": MODEL_FAMILIES.get(name, "Other"),
            "params_m": params_m,
            "trainable_m": trainable_m,
            "input_shape": "(B, 64000)",
            "output_classes": 36,
        })
    return table


def load_pretrained_model(
    model_name: str,
    model: torch.nn.Module,
    checkpoint_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Loads pretrained weights or best checkpoint for the given model if available."""
    dirs_to_check: list[Path] = []
    if checkpoint_dir:
        dirs_to_check.append(Path(checkpoint_dir))
    dirs_to_check.extend([
        PACKAGE_ROOT / "checkpoints",
        PACKAGE_ROOT / "pretrained",
        PACKAGE_ROOT.parent.parent / "pretrained",
        Path("/marimo/review-3/checkpoints"),
        Path("/marimo/pretrained"),
        Path("/datasets/pretrained"),
    ])

    candidate_filenames = [
        f"{model_name}/best_model.pt",
        f"{model_name}/best_checkpoint.pt",
        f"{model_name}_best.pt",
        f"{model_name}.pt",
        f"{model_name}.pth",
    ]
    if model_name == "beats":
        candidate_filenames.extend([
            "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            "BEATs_iter3_plus_AS2M.pt",
            "beat-ft/beats_teacher_fold0_weights.pt",
            "beats/best_model.pt",
        ])
    elif model_name == "pann_cnn10":
        candidate_filenames.extend([
            "Cnn10_mAP=0.380.pth",
            "pann_cnn10/best_model.pt",
        ])
    elif model_name == "ced_tiny":
        candidate_filenames.extend([
            "model.safetensors",
            "ced_tiny/best_model.pt",
        ])
    elif model_name == "efficient_mn10":
        candidate_filenames.extend([
            "mn10_as_mAP_471.pt",
            "efficient_mn10/best_model.pt",
        ])

    for base_dir in dirs_to_check:
        for fname in candidate_filenames:
            p = base_dir / fname
            if p.is_file():
                try:
                    if p.suffix == ".safetensors":
                        from safetensors.torch import load_file
                        state = load_file(str(p))
                    else:
                        state = torch.load(p, map_location="cpu", weights_only=False)

                    sd = state.get("model_state_dict", state.get("model", state))
                    model_sd = model.state_dict()
                    matched = {}
                    for k, v in sd.items():
                        if k in model_sd and model_sd[k].shape == v.shape:
                            matched[k] = v
                        else:
                            k_clean = k.replace("encoder.", "").replace("model.", "").replace("backbone.", "")
                            if k_clean in model_sd and model_sd[k_clean].shape == v.shape and k_clean not in matched:
                                matched[k_clean] = v

                    if len(matched) > 0:
                        model.load_state_dict(matched, strict=False)
                        msg = f"  -> [PRETRAIN] Da nap trong so pretrained cho {model_name} tu {p.name} ({len(matched)}/{len(model_sd)} tensors)"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                        return True
                except Exception as e:
                    if logger:
                        logger.warning(f"  -> [PRETRAIN] Loi khi nap tu {p}: {e}")
    return False


def run_benchmark(
    models: list[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    noise_cache_path: str | None = None,
    data_root: str | None = None,
    smoke_test: bool = False,
    pretrained: bool = True,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
    models_config_path: str | Path | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    Executes benchmark for the 9 models with detailed logging across 6 SNR levels.
    Pulls base hyperparameters from config.yaml and model-specific profiles from models.yaml.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = models or CANONICAL_9_MODELS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load configuration from config.yaml
    config_path = Path(config_path) if config_path else (PACKAGE_ROOT / "config" / "config.yaml")
    cfg: dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 2. Load model-specific configuration from models.yaml
    models_config_path = Path(models_config_path) if models_config_path else (PACKAGE_ROOT / "config" / "models.yaml")
    models_cfg: dict[str, Any] = {}
    if models_config_path.is_file():
        with open(models_config_path, encoding="utf-8") as f:
            models_cfg = yaml.safe_load(f) or {}

    # Resolve defaults from config.yaml if not explicitly provided
    train_cfg_yaml = cfg.get("training", {})
    actual_epochs_target = epochs if epochs is not None else train_cfg_yaml.get("epochs", 50)
    actual_batch_target = batch_size if batch_size is not None else train_cfg_yaml.get("batch_size", 64)
    data_root = data_root or cfg.get("dataset", {}).get("data_root", "/datasets/mix-frcrn")
    noise_cache_path = noise_cache_path or cfg.get("noise_cache", {}).get("cache_dir", data_root)

    log_dir = PACKAGE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"benchmark_9models_{timestamp}.log"
    logger = setup_logger(f"benchmark_9models_{timestamp}", log_file=log_file)

    logger.info("=" * 96)
    logger.info("REVIEW-3: 9-MODEL BENCHMARK PIPELINE")
    logger.info("Timestamp: %s | Device: %s | Epochs: %d (from %s) | Batch Size: %d | Smoke: %s",
                timestamp, device, actual_epochs_target, config_path.name, actual_batch_target, smoke_test)
    logger.info("Config Sources: Main='%s' | Models='%s'", config_path, models_config_path)
    logger.info("Noise Cache: %s | Data Root: %s", noise_cache_path, data_root)
    logger.info("Target Models (%d): %s", len(models), ", ".join(models))
    logger.info("=" * 96)

    # Print Architecture Table
    arch_table = get_models_architecture_table(models)
    logger.info("\n" + "=" * 96)
    logger.info("MODEL ARCHITECTURE & PARAMETERS SUMMARY")
    logger.info("=" * 96)
    logger.info(f"{'#':<3} | {'Model Key':<22} | {'Family':<20} | {'Params (M)':<12} | {'Trainable (M)':<14} | {'Input Shape':<12}")
    logger.info("-" * 96)
    for row in arch_table:
        logger.info(f"{row['idx']:<3} | {row['model_key']:<22} | {row['family']:<20} | {row['params_m']:<12.2f} | {row['trainable_m']:<14.2f} | {row['input_shape']:<12}")
    logger.info("=" * 96 + "\n")

    max_samples = 16 if smoke_test else None
    actual_epochs = 1 if smoke_test else actual_epochs_target
    actual_batch = 4 if smoke_test else actual_batch_target

    # Load shared datasets
    logger.info("Loading Datasets from: %s...", noise_cache_path)
    train_ds = get_dataset(
        root=data_root,
        split="train",
        noise_cache_path=noise_cache_path,
        max_samples=max_samples,
    )
    val_ds = get_dataset(
        root=data_root,
        split="validation",
        noise_cache_path=noise_cache_path,
        max_samples=max_samples,
    )
    test_ds = get_dataset(
        root=data_root,
        split="test",
        noise_cache_path=noise_cache_path,
        max_samples=max_samples,
    )
    logger.info("Datasets Ready: Train=%d, Val=%d, Test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = get_dataloader(train_ds, batch_size=actual_batch, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=actual_batch, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=actual_batch, shuffle=False)

    all_results = []
    snr_levels_list = ["-5.0", "0.0", "5.0", "10.0", "15.0", "20.0"]

    for model_idx, model_name in enumerate(models, 1):
        logger.info("\n" + "=" * 96)
        logger.info(f"[{model_idx}/{len(models)}] TRAINING MODEL: {model_name.upper()}...")
        logger.info("=" * 96)

        start_t = time.time()
        set_seed(2026 + model_idx)
        model = build_model(model_name, num_classes=36)
        total_p, train_p = count_parameters(model)

        # Merge model-specific hyperparameters from models.yaml
        model_profile = get_model_profile(model_name, models_cfg)
        model_cfg = json.loads(json.dumps(cfg))  # deep copy
        model_train_sub = model_cfg.setdefault("training", {})
        model_train_sub["epochs"] = actual_epochs
        model_train_sub["batch_size"] = actual_batch
        if "lr" in model_profile:
            model_train_sub["lr"] = float(model_profile["lr"])
        if "weight_decay" in model_profile:
            model_train_sub["weight_decay"] = float(model_profile["weight_decay"])
        if "warmup_ratio" in model_profile:
            model_train_sub["warmup_ratio"] = float(model_profile["warmup_ratio"])

        logger.info("Model Profile (%s): lr=%s, weight_decay=%s",
                    model_name, model_train_sub.get("lr"), model_train_sub.get("weight_decay"))

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=model_cfg,
            device=device,
            logger=logger,
            checkpoint_dir=PACKAGE_ROOT / "checkpoints",
            model_name=model_name,
        )

        res = trainer.fit()
        elapsed = time.time() - start_t
        res["model_name"] = model_name
        res["params_m"] = round(total_p / 1e6, 2)
        res["elapsed_s"] = round(elapsed, 1)

        # Extract overall and SNR breakdown
        test_data = res.get("test", {})
        overall_metrics = test_data.get("test_metrics", {})
        snr_breakdown = test_data.get("test_snr", {})

        # If test split had no separate data, use best validation epoch
        if not overall_metrics and res.get("history"):
            best_ep = res["history"][-1]
            overall_metrics = best_ep.get("val_metrics", {})
            snr_breakdown = best_ep.get("snr_breakdown", {})

        model_summary = {
            "model_name": model_name,
            "family": MODEL_FAMILIES.get(model_name, "Other"),
            "params_m": round(total_p / 1e6, 2),
            "elapsed_s": round(elapsed, 1),
            "best_epoch": res.get("best_epoch", 1),
            "overall": {
                "loss": round(overall_metrics.get("loss", 0.0), 4),
                "acc": round(overall_metrics.get("acc", 0.0) * 100, 2),
                "precision": round(overall_metrics.get("precision", 0.0), 4),
                "recall": round(overall_metrics.get("recall", 0.0), 4),
                "macro_f1": round(overall_metrics.get("macro_f1", 0.0), 4),
                "micro_f1": round(overall_metrics.get("micro_f1", 0.0), 4),
                "mAP": round(overall_metrics.get("mAP", 0.0), 4),
            },
            "snr_breakdown": {},
        }

        for snr in snr_levels_list:
            snr_m = (
                snr_breakdown.get(snr)
                or snr_breakdown.get(f"{float(snr):.0f}dB")
                or snr_breakdown.get(f"{float(snr):.1f}dB")
                or {}
            )
            model_summary["snr_breakdown"][snr] = {
                "acc": round(snr_m.get("acc", 0.0) * 100, 2),
                "precision": round(snr_m.get("precision", 0.0), 4),
                "recall": round(snr_m.get("recall", 0.0), 4),
                "macro_f1": round(snr_m.get("macro_f1", 0.0), 4),
                "micro_f1": round(snr_m.get("micro_f1", 0.0), 4),
                "mAP": round(snr_m.get("mAP", 0.0), 4),
                "samples": snr_m.get("count", snr_m.get("samples", 0)),
            }

        all_results.append(model_summary)

        if progress_callback:
            progress_callback(model_name, model_idx, len(models), model_summary)

    # 2. Print & Log Unified Comparison Tables across 9 Models
    logger.info("\n" + "=" * 120)
    logger.info("BANG 1: CHI SO TONG THE (ALL METRICS) CUA 9 MO HINH")
    logger.info("=" * 120)
    h1 = (
        f"{'Model Name':<22} | {'Params':<7} | {'Accuracy':<10} | {'Precision':<10} | "
        f"{'Recall':<10} | {'Macro-F1':<10} | {'Micro-F1':<10} | {'mAP':<8}"
    )
    logger.info(h1)
    logger.info("-" * 120)

    for item in sorted(all_results, key=lambda x: x["overall"]["macro_f1"], reverse=True):
        m = item["overall"]
        r1 = (
            f"{item['model_name']:<22} | {item['params_m']:>6.2f}M | {m['acc']:>9.2f}% | "
            f"{m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['macro_f1']:>10.4f} | "
            f"{m['micro_f1']:>10.4f} | {m['mAP']:>8.4f}"
        )
        logger.info(r1)
    logger.info("=" * 120)

    logger.info("\n" + "=" * 138)
    logger.info("BANG 2: CHI TIET THEO 6 MOC SNR (-5dB, 0dB, 5dB, 10dB, 15dB, 20dB)")
    logger.info("=" * 138)
    h2 = (
        f"{'Model Name':<20} | {'-5dB (Acc / F1)':<18} | {'0dB (Acc / F1)':<18} | {'5dB (Acc / F1)':<18} | "
        f"{'10dB (Acc / F1)':<18} | {'15dB (Acc / F1)':<18} | {'20dB (Acc / F1)':<18}"
    )
    logger.info(h2)
    logger.info("-" * 138)

    for item in sorted(all_results, key=lambda x: x["overall"]["macro_f1"], reverse=True):
        s = item["snr_breakdown"]
        def fmt_snr(k: str) -> str:
            d = s.get(k, {})
            return f"{d.get('acc', 0.0):5.1f}% / {d.get('macro_f1', 0.0):.4f}"
        r2 = (
            f"{item['model_name']:<20} | {fmt_snr('-5.0'):<18} | {fmt_snr('0.0'):<18} | "
            f"{fmt_snr('5.0'):<18} | {fmt_snr('10.0'):<18} | {fmt_snr('15.0'):<18} | {fmt_snr('20.0'):<18}"
        )
        logger.info(r2)
    logger.info("=" * 138)

    # 3. Save JSON and latest links
    json_path = log_dir / f"benchmark_9models_{timestamp}.json"
    latest_json = log_dir / "latest_9models_benchmark.json"
    latest_log = log_dir / "latest_9models_benchmark.log"

    report_payload = {
        "timestamp": timestamp,
        "device": str(device),
        "epochs": actual_epochs,
        "batch_size": actual_batch,
        "smoke_test": smoke_test,
        "architecture_summary": arch_table,
        "models_benchmark": all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Copy latest log
    import shutil
    shutil.copyfile(log_file, latest_log)

    logger.info("Log file saved to: %s", log_file)
    logger.info("JSON report saved to: %s", json_path)
    logger.info("Latest benchmark links updated at: %s", latest_json)

    return report_payload


def main():
    parser = argparse.ArgumentParser(description="Review-3: 9-Model Benchmark Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--models-config", type=str, default=None, help="Path to models.yaml (default: config/models.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: config.yaml training.epochs)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: config.yaml training.batch_size)")
    parser.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test on all models")
    parser.add_argument("--noise-cache", type=str, default=None, help="FRCRN cache directory")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset manifest directory")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models or default 9 models")
    args = parser.parse_args()

    run_benchmark(
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_cache_path=args.noise_cache,
        data_root=args.data_root,
        smoke_test=args.smoke,
        config_path=args.config,
        models_config_path=args.models_config,
    )


if __name__ == "__main__":
    main()

