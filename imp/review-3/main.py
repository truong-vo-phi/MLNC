"""
Main Entry-Point CLI for Review-3 Project:
Acoustic Noise Classification across Multiple SNR Levels using 9 Diverse Architectures.

Usage:
  python main.py list-models
  python main.py verify
  python main.py train --model pann_cnn10 [--epochs 30] [--smoke]
  python main.py train --model all [--smoke]
  python main.py eval --model pann_cnn10 [--checkpoint checkpoints/pann_cnn10/best_model.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Safe stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset
from src.metrics import compute_metrics, compute_snr_breakdown
from src.models import build_model, list_available_models
from src.trainer import Trainer
from src.utils import count_parameters, set_seed, setup_logger


def cmd_list_models(args: argparse.Namespace) -> None:
    models_yaml = PACKAGE_ROOT / "config" / "models.yaml"
    with open(models_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("models", {})

    print("=" * 96)
    print("REVIEW-3: 9 ACOUSTIC NOISE CLASSIFICATION MODELS OVERVIEW")
    print("=" * 96)
    print(f"{'#':<3} | {'Model Key':<15} | {'Display Name':<16} | {'Family':<18} | {'Params (M)':<11} | {'Description'}")
    print("-" * 96)

    for idx, key in enumerate(list_available_models(), 1):
        info = cfg.get(key, {})
        disp_name = info.get("display_name", key)
        family = info.get("family", "Unknown")
        desc = info.get("description", "")
        # Compute exact parameters
        try:
            m = build_model(key, num_classes=36)
            total_p, _ = m.count_parameters()
            params_str = f"{total_p / 1e6:6.2f}M"
        except Exception:
            params_str = "N/A"

        print(f"{idx:<3} | {key:<15} | {disp_name:<16} | {family:<18} | {params_str:<11} | {desc}")

    print("=" * 96)


def cmd_verify(args: argparse.Namespace) -> None:
    from scripts.verify_models import verify_all_models
    success = verify_all_models()
    sys.exit(0 if success else 1)


def cmd_train(args: argparse.Namespace) -> None:
    config_path = PACKAGE_ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI Overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["lr"] = args.lr
    if args.data_root:
        config.setdefault("dataset", {})["data_root"] = args.data_root

    device_str = args.device or config.get("training", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    all_keys = list_available_models()
    target_models = all_keys if args.model.lower() == "all" else [args.model]

    for model_name in target_models:
        if model_name not in all_keys:
            print(f"Error: Unknown model '{model_name}'. Available: {all_keys}")
            sys.exit(1)

    print("=" * 86)
    print(f"REVIEW-3: TRAINING {'ALL MODELS' if len(target_models) > 1 else target_models[0].upper()}")
    print(f"Device: {device} | Smoke Test: {args.smoke} | Epochs: {config.get('training', {}).get('epochs')}")
    print("=" * 86)

    from scripts.run_all_models import train_single_model

    results = []
    for model_name in target_models:
        print(f"\n[>>>] Starting training pipeline for: {model_name}...")
        try:
            res = train_single_model(
                model_name=model_name,
                config=config,
                device=device,
                smoke_test=args.smoke,
            )
            results.append(res)
            print(f"[OK] Finished {model_name}: Best Metric = {res['best_metric']:.4f}")
            print(f"    Log file saved to: {res['log_file']}")
        except Exception as e:
            print(f"[ERROR] Failed {model_name}: {e}")
            import traceback
            traceback.print_exc()

    summary_file = PACKAGE_ROOT / "logs" / "latest_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll requested training tasks finished. Summary written to {summary_file}")


def cmd_eval(args: argparse.Namespace) -> None:
    config_path = PACKAGE_ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    ckpt_path = Path(args.checkpoint) if args.checkpoint else PACKAGE_ROOT / "checkpoints" / model_name / "best_model.pt"
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint file not found at: {ckpt_path}")
        sys.exit(1)

    from scripts.evaluate_models import evaluate_model
    data_root = args.data_root or config.get("dataset", {}).get("data_root")
    num_classes = config.get("dataset", {}).get("num_classes", 36)
    noise_cache = getattr(args, "noise_cache", None) or config.get("noise_cache", {}).get("cache_file") or config.get("noise_cache", {}).get("cache_dir")
    apply_rms_norm = not getattr(args, "no_rms_norm", False)
    target_rms = getattr(args, "target_rms", 0.07925)

    evaluate_model(
        model_name=model_name,
        checkpoint_path=ckpt_path,
        data_root=data_root,
        noise_cache_path=noise_cache,
        split=args.split,
        device=device,
        num_classes=num_classes,
        apply_rms_norm=apply_rms_norm,
        target_rms=target_rms,
        max_samples=args.max_samples,
    )


def cmd_eval_masked(args: argparse.Namespace) -> None:
    """Tác vụ 2: Đánh giá mô hình sau khi qua mask FRCRN từ file .npy."""
    from scripts.evaluate_masked_models import evaluate_single_masked_model
    config_path = PACKAGE_ROOT / args.config
    cfg = {}
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_root = args.data_root or cfg.get("dataset", {}).get("data_root")
    num_classes = cfg.get("dataset", {}).get("num_classes", 36)
    noise_cache = args.noise_cache or cfg.get("noise_cache", {}).get("cache_file") or cfg.get("noise_cache", {}).get("cache_dir")
    apply_rms_norm = not args.no_rms_norm
    target_rms = args.target_rms

    device_str = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PACKAGE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_masked_{args.model}_{timestamp}.log"
    logger = setup_logger(f"eval_masked_{timestamp}", log_file=log_file)

    logger.info("=" * 86)
    logger.info("TAC VU 2: DANH GIA MO HINH TREN NOISE FRCRN MASK (.npy)")
    logger.info("Timestamp: %s | Device: %s | Split: %s", timestamp, device, args.split)
    logger.info("Noise Cache File: %s", noise_cache)
    logger.info("RMS Normalization: %s (target_rms = %.5f)", apply_rms_norm, target_rms)
    logger.info("=" * 86)

    models_to_eval = list_available_models() if args.model.lower() == "all" else [args.model]
    results = []

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
            target_rms=target_rms,
            max_samples=args.max_samples,
            allow_untrained=args.allow_untrained,
            logger=logger,
        )
        if res is not None:
            results.append(res)

    if len(results) > 1:
        logger.info("=" * 86)
        logger.info("BANG TONG HOP SO SANH CAC MO HINH SAU KHI QUA MASK FRCRN")
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

    summary_path = log_dir / "eval_masked_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Ket qua danh gia chi tiet da ghi vao: %s", log_file)
    logger.info("File tong hop json: %s", summary_path)
    print(f"\n[OK] Hoan thanh danh gia {len(results)} mo hinh tren tap mask FRCRN (.npy).")
    print(f"     Chi tiet log: {log_file}")


def cmd_kd(args: argparse.Namespace) -> None:
    """Task 3: BEATs -> CED-Tiny Knowledge Distillation Benchmark."""
    from scripts.run_kd_benchmark import run_kd_benchmark
    run_kd_benchmark(args)


def cmd_moe(args: argparse.Namespace) -> None:
    """Task 4: Proposed Single-Backbone MoE (CED-SingleMoE-SplitFFN ~6.62M params) Benchmark."""
    from scripts.run_task4_moe import run_task4_benchmark
    run_task4_benchmark(
        epochs=args.epochs or 15,
        batch_size=args.batch_size or 32,
        smoke=args.smoke,
        no_kd=args.no_kd,
        device_name=None if args.device == "auto" else args.device,
    )


def cmd_kd(args: argparse.Namespace) -> None:
    """Task 3: BEATs -> CED-Tiny Knowledge Distillation Benchmark."""
    from scripts.run_kd_benchmark import run_kd_benchmark
    run_kd_benchmark(args)


def cmd_moe(args: argparse.Namespace) -> None:
    """Task 4: Proposed Single-Backbone MoE (CED-SingleMoE-SplitFFN ~6.62M params) Benchmark."""
    from scripts.run_task4_moe import run_task4_benchmark
    run_task4_benchmark(
        epochs=args.epochs or 15,
        batch_size=args.batch_size or 32,
        smoke=args.smoke,
        no_kd=args.no_kd,
        device_name=None if args.device == "auto" else args.device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Review-3: 9-Model Acoustic Noise Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # list-models
    subparsers.add_parser("list-models", help="List all 9 models with parameters and families")

    # verify
    subparsers.add_parser("verify", help="Run sanity forward-pass verification on all 9 models")

    # train
    parser_train = subparsers.add_parser("train", help="Train a specific model or all models")
    parser_train.add_argument("--model", type=str, required=True, help="Model name (e.g. pann_cnn10, ced_tiny) or 'all'")
    parser_train.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser_train.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_train.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test run")
    parser_train.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_train.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # eval
    parser_eval = subparsers.add_parser("eval", help="Evaluate a trained checkpoint across SNR levels")
    parser_eval.add_argument("--model", type=str, required=True, help="Model name")
    parser_eval.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser_eval.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser_eval.add_argument("--no-rms-norm", action="store_true", help="Disable RMS normalization")
    parser_eval.add_argument("--target-rms", type=float, default=0.07925, help="Target RMS value (default: 0.07925)")
    parser_eval.add_argument("--split", type=str, default="validation", help="Dataset split (validation or test)")
    parser_eval.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser_eval.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_eval.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # eval-masked (Task 2: Evaluate models on FRCRN-masked noise from .npy)
    parser_eval_masked = subparsers.add_parser("eval-masked", help="Task 2: Evaluate models on FRCRN-masked noise (.npy)")
    parser_eval_masked.add_argument("--model", type=str, default="all", help="Model name (e.g. pann_cnn10, ced_tiny) or 'all'")
    parser_eval_masked.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy file or folder")
    parser_eval_masked.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser_eval_masked.add_argument("--split", type=str, default="validation", help="Dataset split (validation, test, train)")
    parser_eval_masked.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser_eval_masked.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser_eval_masked.add_argument("--no-rms-norm", action="store_true", help="Disable RMS normalization")
    parser_eval_masked.add_argument("--target-rms", type=float, default=0.07925, help="Target RMS (default: 0.07925)")
    parser_eval_masked.add_argument("--allow-untrained", action="store_true", help="Evaluate untrained model baseline if checkpoint missing")
    parser_eval_masked.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_eval_masked.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset for manifest")
    parser_eval_masked.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # kd (Task 3: BEATs -> CED-Tiny Knowledge Distillation)
    parser_kd = subparsers.add_parser("kd", help="Task 3: BEATs -> CED-Tiny KD (5 methods + BeatBiasPostLN comparison)")
    parser_kd.add_argument("--method", type=str, default="all",
                           choices=["all", "logit", "crd", "mgd", "logit+crd", "logit+mgd"],
                           help="Specific KD method or 'all' to run complete benchmark")
    parser_kd.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser_kd.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser_kd.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test")
    parser_kd.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser_kd.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_kd.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_kd.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # moe (Task 4: Proposed CED-SingleMoE-SplitFFN ~6.62M params Benchmark)
    parser_moe = subparsers.add_parser("moe", help="Task 4: Benchmark proposed CED-SingleMoE-SplitFFN (~6.62M params)")
    parser_moe.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser_moe.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser_moe.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test")
    parser_moe.add_argument("--no-kd", action="store_true", help="Disable BEATs Knowledge Distillation")
    parser_moe.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")

    # kd (Task 3: BEATs -> CED-Tiny Knowledge Distillation)
    parser_kd = subparsers.add_parser("kd", help="Task 3: BEATs -> CED-Tiny KD (5 methods + BeatBiasPostLN comparison)")
    parser_kd.add_argument("--method", type=str, default="all",
                           choices=["all", "logit", "crd", "mgd", "logit+crd", "logit+mgd"],
                           help="Specific KD method or 'all' to run complete benchmark")
    parser_kd.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser_kd.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser_kd.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test")
    parser_kd.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser_kd.add_argument("--data-root", type=str, default=None, help="Root path of mixture dataset")
    parser_kd.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser_kd.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    # moe (Task 4: Proposed CED-SingleMoE-SplitFFN ~6.62M params Benchmark)
    parser_moe = subparsers.add_parser("moe", help="Task 4: Benchmark proposed CED-SingleMoE-SplitFFN (~6.62M params)")
    parser_moe.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser_moe.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser_moe.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test")
    parser_moe.add_argument("--no-kd", action="store_true", help="Disable BEATs Knowledge Distillation")
    parser_moe.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")

    args = parser.parse_args()

    if args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "eval-masked":
        cmd_eval_masked(args)
    elif args.command == "kd":
        cmd_kd(args)
    elif args.command == "moe":
        cmd_moe(args)
    elif args.command == "kd":
        cmd_kd(args)
    elif args.command == "moe":
        cmd_moe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
