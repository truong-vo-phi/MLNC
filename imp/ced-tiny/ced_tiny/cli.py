from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model.frcrn import FRCRNSeparator

from .config import PACKAGE_ROOT, load_config, resolve_path
from .evaluate import collect_predictions, compute_metrics, compute_snr_breakdown
from .frontend import LogMel, SpecAugment
from .model import CEDTiny
from .pipeline import build_datasets, build_loaders, load_separator, run_pipeline
from .runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CED-Tiny: SED với kiến trúc CED-Tiny qua bộ tách nhiễu FRCRN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_cmd = subparsers.add_parser("train", help="Huấn luyện CED-Tiny đầy đủ")
    train_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ced_tiny.yaml"))
    train_cmd.add_argument("--run-id", type=str, default=None)
    train_cmd.add_argument("--output-dir", type=str, default=None)
    train_cmd.add_argument("--device", type=str, default=None)

    smoke_cmd = subparsers.add_parser("smoke", help="Chạy smoke test nhanh kiểm tra pipeline")
    smoke_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ced_tiny.yaml"))
    smoke_cmd.add_argument("--output-dir", type=str, default=None)
    smoke_cmd.add_argument("--device", type=str, default=None)

    params_cmd = subparsers.add_parser("params", help="Kiểm tra tham số của FRCRN và CED-Tiny")
    params_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ced_tiny.yaml"))

    eval_cmd = subparsers.add_parser("evaluate", help="Đánh giá checkpoint có sẵn")
    eval_cmd.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "ced_tiny.yaml"))
    eval_cmd.add_argument("--checkpoint", type=str, required=True)
    eval_cmd.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    eval_cmd.add_argument("--device", type=str, default=None)

    export_cmd = subparsers.add_parser("export", help="Xuất checkpoint CED-Tiny độc lập")
    export_cmd.add_argument("--checkpoint", type=str, required=True)
    export_cmd.add_argument("--output", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "params":
        cfg = load_config(args.config)
        print("=" * 65)
        print("CED-Tiny Pipeline Configuration & Parameters Summary")
        print("=" * 65)
        print(f"FRCRN config:        {cfg.frcrn_config}")
        print(f"FRCRN Checkpoint:    {cfg.frcrn.checkpoint}")
        print(f"Target RMS:          {cfg.frcrn.target_rms}")
        print(f"CED Architecture:    {cfg.ced_tiny.arch} (init={cfg.ced_tiny.init})")
        print(f"CED Positional Bias: {cfg.ced_tiny.bias_init} (T5 buckets + GRU gate)")
        print(f"Frontend:            LogMel ({cfg.frontend.n_mels} mels, {cfg.frontend.num_frames} frames)")
        print(f"Learning Rate:       {cfg.train.lr}")
        print(f"Batch Size:          {cfg.train.batch_size} (acc={cfg.train.accumulation_steps})")
        print(f"Epochs:              {cfg.train.epochs}")
        print(f"Selection Metric:    {cfg.train.selection_metric}")

        frontend = LogMel(**cfg.frontend.to_dict())
        spec_augment = SpecAugment(**cfg.specaugment.to_dict())
        ced_cfg_dict = cfg.ced_tiny.to_dict()
        ced_cfg_dict["init"] = "none"
        from .config import CEDTinyConfig

        ced = CEDTiny(CEDTinyConfig.from_dict(ced_cfg_dict), frontend, spec_augment)
        print("-" * 65)
        print(f"CED-Tiny Total Parameters:     {ced.num_parameters(trainable_only=False) / 1e6:.2f}M")
        print(f"CED-Tiny Trainable Parameters: {ced.num_parameters(trainable_only=True) / 1e6:.2f}M")
        groups = ced.describe_params()
        for g_name, g_count in groups.items():
            print(f"  - {g_name:<35}: {g_count:>10,d} ({g_count / 1e6:.3f}M)")

        frcrn_cfg_path = resolve_path(cfg.frcrn_config)
        if frcrn_cfg_path.is_file():
            frcrn = FRCRNSeparator.from_yaml(frcrn_cfg_path)
            print("-" * 65)
            print(f"FRCRN Parameters (Frozen):     {frcrn.num_parameters(trainable_only=False) / 1e6:.2f}M")
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
        classifier = CEDTiny(
            cfg.ced_tiny,
            LogMel(**cfg.frontend.to_dict()),
            SpecAugment(**cfg.specaugment.to_dict()),
        )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        classifier.load_state_dict(state_dict)
        classifier.to(dev).eval()

        datasets = build_datasets(cfg.data)
        split = args.split
        if split not in datasets:
            raise ValueError(f"Split {split} not available in dataset")
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
            loader=loaders[split],
            device=dev,
            separator=separator,
            target_rms=cfg.frcrn.target_rms,
            amp_mode=cfg.train.amp,
            desc=f"{split}_eval",
        )
        overall = compute_metrics(preds.probs, preds.labels, classifier.config.num_classes)
        by_snr = compute_snr_breakdown(preds, cfg.data.snr_levels, classifier.config.num_classes)

        print("=" * 60)
        print(f"Evaluation on '{split}' split ({len(preds)} samples)")
        print("=" * 60)
        for k, v in overall.items():
            print(f"  {k:<20}: {v:.4f}")
        print("-" * 60)
        print("SNR Breakdown:")
        levels: set[int] = set()
        for k in by_snr:
            if k.startswith("acc_snr_"):
                levels.add(int(float(k.replace("acc_snr_", ""))))
        for lvl in sorted(levels):
            acc = by_snr.get(f"acc_snr_{lvl}", 0.0)
            f1 = by_snr.get(f"macro_f1_snr_{lvl}", 0.0)
            print(f"  SNR {lvl:>3} dB: Acc = {acc:.4f} | F1 = {f1:.4f}")
        print("=" * 60)
        return

    if args.command == "export":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model", payload)
        torch.save({"model": state_dict}, out_path)
        print(f"Exported standalone CED-Tiny model weights to {out_path}")
