"""
Training Runner Script for Dual-CED-Tiny Models:
Supports full 15-epoch training and smoke test verification on synthetic or real data.
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

from src.dataset import get_dataloader, get_dataset
from src.kd_trainer import KDTrainer
from src.models import build_model
from src.trainer import Trainer
from src.utils import set_seed, setup_logger


def _build_teacher_logits(kd_cfg, train_dataset, val_dataset, device, logger, use_synthetic):
    """Teacher logits [N, 2, C] for the train split: index 0 = fold0 (Low expert), 1 = EMA (High expert)."""
    import numpy as np

    from src.teachers import compute_teacher_logits, load_beats_teacher, resolve_teacher_path

    num_classes = getattr(train_dataset, "num_classes", 36)
    if use_synthetic:
        # Synthetic smoke test: no real audio, so teachers are replaced by random logits (pipeline check only).
        logger.warning("Synthetic data: using RANDOM teacher logits (smoke test of the KD pipeline only).")
        rng = np.random.default_rng(2026)
        return rng.standard_normal((len(train_dataset), 2, num_classes)).astype("float32")

    cache_dir = ROOT / kd_cfg.get("cache_dir", "logs/teacher_cache")
    beat_cfg = ROOT / kd_cfg.get("beat_config", "config/beat.yaml")
    bs = int(kd_cfg.get("teacher_batch_size", 32))
    input_rms = float(kd_cfg.get("input_rms", 0.07925))
    per_teacher = []
    for tag, key in (("low_fold0", "low"), ("high_ema", "high")):
        spec = kd_cfg["teachers"][key]
        ckpt = resolve_teacher_path(spec.get("path"), spec.get("hf_file"), kd_cfg.get("hf_repo"), ROOT)
        teacher, info = load_beats_teacher(ckpt, beat_cfg, device)
        logger.info("[teacher:%s] %s (%s): %s params (%.2fM), frozen, inference only",
                    tag, ckpt.name, info["format"], f"{info['params']:,}", info["params"] / 1e6)
        train_logits = compute_teacher_logits(
            teacher,
            train_dataset,
            bs,
            device,
            cache_dir / f"train_{tag}_{len(train_dataset)}_rms{input_rms:.5f}.npy",
            input_rms=input_rms,
        )
        val_logits = compute_teacher_logits(
            teacher,
            val_dataset,
            bs,
            device,
            cache_dir / f"val_{tag}_{len(val_dataset)}_rms{input_rms:.5f}.npy",
            input_rms=input_rms,
        )
        if hasattr(val_dataset, "targets"):
            acc = float((val_logits.argmax(-1) == val_dataset.targets).mean())
            logger.info("[teacher:%s] val accuracy on FRCRN input: %.2f%%", tag, acc * 100)
        per_teacher.append(train_logits)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(per_teacher, axis=1)


def train_model(
    model_name: str = "dual_ced_tiny",
    config_path: Path | str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    smoke: bool = False,
    device_str: str | None = None,
    use_synthetic: bool = False,
    no_kd: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path) if config_path else (ROOT / "config" / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.setdefault("training", {})
    if epochs:
        train_cfg["epochs"] = epochs
    if batch_size:
        train_cfg["batch_size"] = batch_size

    if smoke:
        train_cfg["epochs"] = 1
        use_synthetic = True

    seed = train_cfg.get("seed", 2026)
    set_seed(seed)

    if device_str:
        device = torch.device(device_str)
    else:
        dev_cfg = train_cfg.get("device", "auto")
        device = torch.device("cuda" if (dev_cfg == "cuda" or (dev_cfg == "auto" and torch.cuda.is_available())) else "cpu")

    logger = setup_logger(name=model_name, log_dir=ROOT / config.get("logging", {}).get("log_dir", "logs"))

    # Dataloaders
    max_samples = 64 if smoke else None
    train_dataset = get_dataset(config, split="train", use_synthetic=use_synthetic, max_samples=max_samples)
    val_dataset = get_dataset(config, split="eval", use_synthetic=use_synthetic, max_samples=max_samples)

    bs = train_cfg.get("batch_size", 32)
    train_loader = get_dataloader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=bs, shuffle=False)

    # Build model
    router_cfg = config.get("router", {})
    architecture_cfg = config.get("architecture", {}) or {}
    model = build_model(
        model_name=model_name,
        num_classes=config.get("dataset", {}).get("num_classes", 36),
        sample_rate=config.get("dataset", {}).get("sample_rate", 16000),
        n_mels=config.get("frontend", {}).get("n_mels", 64),
        num_frames=config.get("frontend", {}).get("num_frames", 398),
        router_transition_center_db=float(router_cfg.get("transition_center_db", 15.0)),
        router_transition_tau=float(router_cfg.get("transition_tau_db", 2.5)),
        router_token_proj_dim=int(router_cfg.get("token_proj_dim", 16)),
        router_hidden_dim=int(router_cfg.get("hidden_dim", 32)),
        use_grep=bool(architecture_cfg.get("use_grep", True)),
        post_ln=bool(architecture_cfg.get("post_ln", True)),
        pre_ln=bool(architecture_cfg.get("pre_ln", True)),
    )

    if train_cfg.get("pretrained", True) and not use_synthetic:
        pretrained_src = train_cfg.get("pretrained_source", "mispeech/ced-tiny")
        min_tensors = int(train_cfg.get("pretrained_min_tensors", 100))
        try:
            if hasattr(model, "load_pretrained_backbones"):
                n_low, n_high = model.load_pretrained_backbones(pretrained_src)
                if min(n_low, n_high) < min_tensors:
                    raise RuntimeError(
                        f"Only {n_low}/{n_high} pretrained tensors loaded (expected >= {min_tensors} per expert)."
                    )
                logger.info("[*] CED-Tiny pretrained loaded (%s): low_expert=%d tensors, high_expert=%d tensors",
                            pretrained_src, n_low, n_high)
            else:
                from src.models import load_ced_pretrained
                n_loaded = load_ced_pretrained(model, pretrained_src)
                if n_loaded < min_tensors:
                    raise RuntimeError(
                        f"Only {n_loaded} pretrained tensors loaded (expected >= {min_tensors})."
                    )
                logger.info("[*] CED-Tiny pretrained loaded (%s): %d tensors", pretrained_src, n_loaded)
        except Exception as exc:
            raise RuntimeError(
                f"Pretrained CED-Tiny weights were requested (training.pretrained=true) but could not be loaded "
                f"from '{pretrained_src}'. Refusing to train the backbones from scratch silently. "
                f"Fix the checkpoint path/network access or set training.pretrained=false explicitly."
            ) from exc
    elif train_cfg.get("pretrained", True) and use_synthetic:
        logger.warning("Synthetic smoke run: skipping CED-Tiny pretrained weights (real runs require them).")

    trainer_kwargs = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=ROOT / config.get("logging", {}).get("checkpoint_dir", "checkpoints"),
        model_name=model_name,
    )

    kd_cfg = config.get("kd", {}) or {}
    use_kd = kd_cfg.get("enabled", False) and not no_kd and model_name == "dual_ced_tiny"
    if use_kd:
        teacher_logits = _build_teacher_logits(kd_cfg, train_dataset, val_dataset, device, logger, use_synthetic)
        trainer = KDTrainer(teacher_logits_train=teacher_logits, kd_config=kd_cfg, **trainer_kwargs)
    else:
        logger.info("Knowledge distillation disabled: using fused CE, expert CE, and learned-router BCE when supported.")
        trainer = Trainer(**trainer_kwargs)

    result = trainer.fit()

    hf_cfg = config.get("high_expert_finetune", {}) or {}
    if use_kd and not use_synthetic and hf_cfg.get("enabled", False):
        result["high_expert_finetune"] = trainer.finetune_high_expert(
            epochs=int(hf_cfg.get("epochs", 6)),
            lr=float(hf_cfg.get("lr", 5.0e-5)),
            snr_db=float(hf_cfg.get("snr_db", 18.0)),
            patience=int(hf_cfg.get("patience", 3)),
        )

    out_file = ROOT / config.get("logging", {}).get("log_dir", "logs") / f"{model_name}_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved summary report to %s", out_file)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dual-CED-Tiny Models")
    parser.add_argument("--model", default="dual_ced_tiny", choices=["dual_ced_tiny", "ced_tiny_low", "ced_tiny_high"])
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch size")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on synthetic samples")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset")
    parser.add_argument("--no-kd", action="store_true", help="Disable KD (plain cross-entropy)")
    parser.add_argument("--device", default=None, help="Computation device ('cpu' or 'cuda')")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
