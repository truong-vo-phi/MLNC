from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.model.beat import BEATClassifier

from .config import PACKAGE_ROOT, REPO_ROOT, load_config, smoke_config
from .pipeline import run_pipeline
from .runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMA-BEAT: Fine-tuning BEATs với Exponential Moving Average")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Train command
    train_cmd = subparsers.add_parser("train", help="Huấn luyện BEATs với EMA đầy đủ")
    train_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ema_beat.yaml"))
    train_cmd.add_argument("--run-id", type=str, default=None, help="ID của lần chạy nếu muốn tiếp tục (resume)")
    train_cmd.add_argument("--output-dir", type=str, default=None)
    train_cmd.add_argument("--device", type=str, default=None)

    # 2. Smoke test command
    smoke_cmd = subparsers.add_parser("smoke", help="Chạy smoke test nhanh kiểm tra pipeline")
    smoke_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ema_beat.yaml"))
    smoke_cmd.add_argument("--output-dir", type=str, default=None)
    smoke_cmd.add_argument("--device", type=str, default=None)

    # 3. Params command
    params_cmd = subparsers.add_parser("params", help="Kiểm tra thông số mô hình và cấu hình EMA")
    params_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ema_beat.yaml"))

    # 4. Export command
    export_cmd = subparsers.add_parser("export", help="Bóc tách trọng số EMA thành checkpoint độc lập")
    export_cmd.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file resume.pt hoặc best checkpoint")
    export_cmd.add_argument("--output", type=str, required=True, help="Đường dẫn file .pt xuất ra")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "params":
        cfg = load_config(args.config)
        print("=" * 60)
        print("EMA-BEATs Configuration Summary")
        print("=" * 60)
        print(f"Model config:        {cfg.model_config}")
        print(f"EMA Decay:           {cfg.ema.decay}")
        print(f"EMA Warmup Steps:    {cfg.ema.warmup_steps}")
        print(f"EMA Warmup Start:    {cfg.ema.warmup_decay_start}")
        print(f"Selection Target:    {cfg.ema.selection_target}")
        print(f"Learning Rate:       {cfg.train.lr}")
        print(f"Batch Size:          {cfg.train.batch_size}")
        print(f"Epochs:              {cfg.train.epochs}")
        print(f"Selection Metric:    {cfg.train.selection_metric}")

        model_cfg_path = REPO_ROOT / cfg.model_config
        if model_cfg_path.is_file():
            beats = BEATClassifier.from_yaml(model_cfg_path)
            total_params = beats.num_parameters(trainable_only=False)
            trainable_params = beats.num_parameters(trainable_only=True)
            print("-" * 60)
            print(f"BEATs Total Parameters:     {total_params / 1e6:.2f}M")
            print(f"BEATs Trainable Parameters: {trainable_params / 1e6:.2f}M")
            print(f"Shadow EMA Model Memory:    ~{(total_params * 4) / (1024**2):.1f} MB (float32)")
        print("=" * 60)
        return

    if args.command in ("train", "smoke"):
        cfg = load_config(args.config)
        out_dir = Path(args.output_dir) if args.output_dir else None
        dev = resolve_device(args.device)
        is_smoke = args.command == "smoke"
        run_id = getattr(args, "run_id", None)
        run_pipeline(cfg, output_dir=out_dir, run_id=run_id, smoke=is_smoke, device=dev)
        return

    if args.command == "export":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if "ema" in data and data["ema"] is not None:
            ema_state = data["ema"]["ema_model"]
            torch.save({"model": ema_state}, out_path)
            print(f"Successfully exported EMA weights to {out_path}")
        elif "model" in data:
            torch.save({"model": data["model"]}, out_path)
            print(f"Successfully exported model weights to {out_path}")
        else:
            torch.save(data, out_path)
            print(f"Successfully copied weights to {out_path}")


if __name__ == "__main__":
    main()
