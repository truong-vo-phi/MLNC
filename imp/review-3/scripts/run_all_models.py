"""
Batch Training Runner:
Trains each of the 9 models sequentially, saves checkpoints and .log files,
and compiles a unified comparative benchmark table across all models and SNR levels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset
from src.models import build_model, list_available_models
from src.trainer import Trainer
from src.utils import count_parameters, set_seed, setup_logger


def train_single_model(
    model_name: str,
    config: dict,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
    smoke_test: bool = False,
    noise_cache_path: str | Path | None = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PACKAGE_ROOT / config.get("logging", {}).get("log_dir", "logs")
    log_file = log_dir / f"{model_name}_{timestamp}.log"
    logger = setup_logger(name=f"train_{model_name}", log_file=log_file)

    logger.info("Initializing run for model: %s (Log file: %s)", model_name, log_file)

    data_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    if epochs:
        train_cfg["epochs"] = epochs
    if batch_size:
        train_cfg["batch_size"] = batch_size

    if smoke_test:
        train_cfg["epochs"] = 1
        train_cfg["batch_size"] = 4
        max_samples = 8
    else:
        max_samples = None

    set_seed(train_cfg.get("seed", 42))

    data_root = data_cfg.get("data_root")
    num_classes = data_cfg.get("num_classes", 36)
    noise_cache = (
        noise_cache_path
        or config.get("noise_cache", {}).get("cache_dir")
        or config.get("noise_cache", {}).get("cache_file")
        or data_cfg.get("noise_cache_path")
    )

    train_ds = get_dataset(
        root=data_root,
        split="train",
        num_classes=num_classes,
        max_samples=max_samples,
        noise_cache_path=noise_cache,
    )
    val_ds = get_dataset(
        root=data_root,
        split="validation",
        num_classes=num_classes,
        max_samples=max_samples,
        noise_cache_path=noise_cache,
    )
    test_ds = get_dataset(
        root=data_root,
        split="test",
        num_classes=num_classes,
        max_samples=max_samples,
        noise_cache_path=noise_cache,
    )

    train_loader = get_dataloader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=train_cfg.get("eval_batch_size", 32), shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=train_cfg.get("eval_batch_size", 32), shuffle=False)

    model = build_model(model_name, num_classes=num_classes)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=PACKAGE_ROOT / "checkpoints",
        model_name=model_name,
    )

    result = trainer.fit()
    result["model_name"] = model_name
    result["log_file"] = str(log_file)
    total_p, _ = count_parameters(model)
    result["params_m"] = total_p / 1e6
    return result


def main():
    parser = argparse.ArgumentParser(description="Train all Review-3 models")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-epoch smoke test on all models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg_path = PACKAGE_ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_models = list_available_models()
    selected_models = all_models if "all" in args.models else args.models

    print("=" * 86)
    print(f"REVIEW-3: RUNNING BATCH TRAINING FOR {len(selected_models)} MODELS (Smoke={args.smoke})")
    print(f"Device: {device} | Models: {', '.join(selected_models)}")
    print("=" * 86)

    summary_results = []
    for model_name in selected_models:
        print(f"\n>>> Starting model: {model_name}...")
        start_t = time.time()
        try:
            res = train_single_model(
                model_name,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                smoke_test=args.smoke,
                noise_cache_path=args.noise_cache,
            )
            elapsed = time.time() - start_t
            res["elapsed_s"] = elapsed
            summary_results.append(res)
            print(f">>> Finished {model_name} in {elapsed:.1f}s | Best Metric: {res['best_metric']:.4f}")
        except Exception as e:
            print(f">>> FAILED {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary report
    summary_path = PACKAGE_ROOT / "logs" / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nBatch training completed! Benchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()

