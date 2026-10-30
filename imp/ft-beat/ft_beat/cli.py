from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model.beat import BEATClassifier
from src.model.frcrn import FRCRNSeparator

from .config import PACKAGE_ROOT, load_config, resolve_path
from .evaluate import collect_predictions, compute_metrics, compute_snr_breakdown
from .pipeline import build_datasets, build_loaders, load_separator, run_pipeline
from .runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FT-BEAT: Fine-tuning BEATs trên hỗn hợp âm thanh qua FRCRN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_cmd = subparsers.add_parser("train", help="Huấn luyện BEATs đầy đủ")
    train_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ft_beat.yaml"))
    train_cmd.add_argument("--run-id", type=str, default=None)
    train_cmd.add_argument("--output-dir", type=str, default=None)
    train_cmd.add_argument("--device", type=str, default=None)

    smoke_cmd = subparsers.add_parser("smoke", help="Chạy smoke test nhanh kiểm tra pipeline")
    smoke_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ft_beat.yaml"))
    smoke_cmd.add_argument("--output-dir", type=str, default=None)
    smoke_cmd.add_argument("--device", type=str, default=None)

    params_cmd = subparsers.add_parser("params", help="Kiểm tra tham số của FRCRN và BEATs")
    params_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ft_beat.yaml"))

    eval_cmd = subparsers.add_parser("evaluate", help="Đánh giá checkpoint có sẵn")
    eval_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ft_beat.yaml"))
    eval_cmd.add_argument("--checkpoint", type=str, required=True)
    eval_cmd.add_argument("--device", type=str, default=None)

    export_cmd = subparsers.add_parser("export", help="Xuất checkpoint BEATs độc lập hoặc combined pipeline")
    export_cmd.add_argument("--checkpoint", type=str, required=True)
    export_cmd.add_argument("--output", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "params":
        cfg = load_config(args.config)
        print("=" * 65)
        print("FT-BEAT Configuration Summary")
        print("=" * 65)
        print(f"Model config:        {cfg.model_config}")
        print(f"FRCRN config:        {cfg.frcrn_config}")
        print(f"FRCRN Checkpoint:    {cfg.frcrn.checkpoint}")
        print(f"Target RMS:          {cfg.frcrn.target_rms}")
        print(f"Learning Rate:       {cfg.train.lr}")
        print(f"Batch Size:          {cfg.train.batch_size} (acc={cfg.train.accumulation_steps})")
        print(f"Effective Batch:     {cfg.train.batch_size * cfg.train.accumulation_steps}")
        print(f"Epochs:              {cfg.train.epochs}")
        print(f"Selection Metric:    {cfg.train.selection_metric}")

        model_cfg_path = resolve_path(cfg.model_config)
        frcrn_cfg_path = resolve_path(cfg.frcrn_config)
        if model_cfg_path.is_file():
            beats = BEATClassifier.from_yaml(model_cfg_path)
            print("-" * 65)
            print(f"BEATs Total Parameters:     {beats.num_parameters(trainable_only=False) / 1e6:.2f}M")
            print(f"BEATs Trainable Parameters: {beats.num_parameters(trainable_only=True) / 1e6:.2f}M")
        if frcrn_cfg_path.is_file():
            frcrn = FRCRNSeparator.from_yaml(frcrn_cfg_path)
            print(f"FRCRN Parameters (Frozen):  {frcrn.num_parameters(trainable_only=False) / 1e6:.2f}M")
        print("=" * 65)
        return

    if args.command in ("train", "smoke"):
        cfg = load_config(args.config)
        out_dir = Path(args.output_dir) if args.output_dir else None
        dev = resolve_device(args.device)
        is_smoke = args.command == "smoke"
        run_id = getattr(args, "run_id", None)
        run_pipeline(cfg, output_dir=out_dir, run_id=run_id, smoke=is_smoke, device=dev)
        return

    if args.command == "evaluate":
        cfg = load_config(args.config)
        dev = resolve_device(args.device)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        separator = load_separator(cfg, dev, None)
        classifier = BEATClassifier.from_yaml(resolve_path(cfg.model_config))
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        classifier.load_state_dict(state_dict)
        classifier.to(dev).eval()

        datasets = build_datasets(cfg.data)
        loaders = build_loaders(
            datasets,
            batch_size=cfg.train.batch_size,
            eval_batch_size=cfg.train.eval_batch_size,
            num_workers=cfg.data.num_workers,
            balanced_sampling=False,
            seed=cfg.train.seed,
        )

        preds = collect_predictions(
            model=classifier,
            loader=loaders["validation"],
            device=dev,
            separator=separator,
            target_rms=cfg.frcrn.target_rms,
            amp_mode=cfg.train.amp,
            desc="evaluate",
        )
        overall = compute_metrics(preds.probs, preds.labels, classifier.config.num_classes)
        by_snr = compute_snr_breakdown(preds, cfg.data.snr_levels, classifier.config.num_classes)

        print("\n" + "=" * 65)
        print("Validation Results:")
        for k, v in overall.items():
            print(f"  {k:22s}: {v:.4f}")
        print("-" * 65)
        print("Per-SNR Accuracy & Macro-F1:")
        for snr_k, snr_v in by_snr.items():
            print(
                f"  SNR {snr_k:5s} dB | Acc: {snr_v.get('top1_accuracy', 0.0):.4f} | F1: {snr_v.get('macro_f1', 0.0):.4f}"
            )
        print("=" * 65 + "\n")
        return

    if args.command == "export":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model_state = data.get("model", data)
        torch.save({"model": model_state}, out_path)
        print(f"Successfully exported model checkpoint to {out_path}")
