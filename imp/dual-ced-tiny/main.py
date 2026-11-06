"""
Main CLI Entry-Point for Dual-CED-Tiny:
Standalone Dual-Expert Audio Classifier (GREP + Alpha Post-LN + learned router; ~10.80M Parameters).

Usage:
  python main.py list-models
  python main.py verify
  python main.py train [--model dual_ced_tiny] [--epochs 15] [--smoke]
  python main.py eval [--model dual_ced_tiny] [--checkpoint checkpoints_moev2_router/dual_ced_tiny/best_model.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Safe stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import yaml
from src.models import build_model, list_available_models
from src.utils import count_parameters


def cmd_list_models(args: argparse.Namespace) -> None:
    models_yaml = PACKAGE_ROOT / "config" / "models.yaml"
    cfg = {}
    if models_yaml.is_file():
        with open(models_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f).get("models", {})

    print("=" * 100)
    print("DUAL-CED-TINY: ARCHITECTURE & EXPERT OVERVIEW")
    print("=" * 100)
    print(f"{'#':<3} | {'Model Key':<15} | {'Display Name':<22} | {'Family':<22} | {'Params (M)':<11} | {'Description'}")
    print("-" * 100)

    for idx, key in enumerate(list_available_models(), 1):
        info = cfg.get(key, {})
        disp_name = info.get("display_name", key)
        family = info.get("family", "Audio-Transformer")
        desc = info.get("description", "")

        try:
            m = build_model(key, num_classes=36)
            total_p, _ = count_parameters(m)
            params_str = f"{total_p / 1e6:6.2f}M"
        except Exception:
            params_str = "N/A"

        print(f"{idx:<3} | {key:<15} | {disp_name:<22} | {family:<22} | {params_str:<11} | {desc}")

    print("=" * 100)


def cmd_verify(args: argparse.Namespace) -> None:
    from scripts.verify_models import verify_all_models
    success = verify_all_models()
    sys.exit(0 if success else 1)


def cmd_train(args: argparse.Namespace) -> None:
    from scripts.run_training import train_model

    print("=" * 86)
    print(f"DUAL-CED-TINY: TRAINING PIPELINE [{args.model.upper()}]")
    print(f"Epochs: {args.epochs or 15} | Smoke Test: {args.smoke} | Synthetic: {args.synthetic}")
    print("=" * 86)

    train_model(
        model_name=args.model,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoke=args.smoke,
        device_str=args.device,
        use_synthetic=args.synthetic,
        no_kd=args.no_kd,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    from scripts.evaluate_models import evaluate_model

    print("=" * 86)
    print(f"DUAL-CED-TINY: EVALUATION PIPELINE [{args.model.upper()}]")
    print(f"Checkpoint: {args.checkpoint or 'config logging.checkpoint_dir/' + args.model + '/best_model.pt'}")
    print("=" * 86)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual-ced-tiny",
        description="Dual-CED-Tiny Audio Classification System (GREP + Alpha Post-LN + learned router; ~10.80M Parameters)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. list-models
    subparsers.add_parser("list-models", help="List registered models and parameter counts")

    # 2. verify
    subparsers.add_parser("verify", help="Verify model shapes, parameter counts and routing dynamics")

    # 3. train
    p_train = subparsers.add_parser("train", help="Run training pipeline")
    p_train.add_argument("--model", default="dual_ced_tiny", choices=["dual_ced_tiny", "ced_tiny_low", "ced_tiny_high"])
    p_train.add_argument("--config", default=None, help="Path to config.yaml")
    p_train.add_argument("--epochs", type=int, default=None, help="Total epochs")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p_train.add_argument("--smoke", action="store_true", help="Run 1-epoch quick verification on synthetic data")
    p_train.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    p_train.add_argument("--no-kd", action="store_true", help="Disable KD (plain cross-entropy)")
    p_train.add_argument("--device", default=None, help="Device ('cpu' or 'cuda')")

    # 4. eval
    p_eval = subparsers.add_parser("eval", help="Evaluate model checkpoint")
    p_eval.add_argument("--model", default="dual_ced_tiny", choices=["dual_ced_tiny", "ced_tiny_low", "ced_tiny_high"])
    p_eval.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt")
    p_eval.add_argument("--config", default=None, help="Path to config.yaml")
    p_eval.add_argument("--split", default="eval", help="Evaluation split ('eval' or 'test')")
    p_eval.add_argument("--batch-size", type=int, default=32)
    p_eval.add_argument("--synthetic", action="store_true", help="Force synthetic evaluation")
    p_eval.add_argument("--device", default=None)
    p_eval.add_argument("--overlap", dest="overlap", action="store_true", help="Force multi-crop overlap inference")
    p_eval.add_argument("--no-overlap", dest="overlap", action="store_false", help="Disable multi-crop overlap inference")
    p_eval.set_defaults(overlap=None)
    p_eval.add_argument("--num-windows", type=int, default=None, help="Number of overlapping crops (default from config)")
    p_eval.add_argument("--crop-seconds", type=float, default=None, help="Crop duration in seconds (default from config)")
    p_eval.add_argument("--aggregate", default=None, choices=["logits", "probs"], help="Overlap aggregation")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "list-models": cmd_list_models,
        "verify": cmd_verify,
        "train": cmd_train,
        "eval": cmd_eval,
    }
    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
