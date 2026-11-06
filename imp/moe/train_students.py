from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.load_dataset import build_cached_loaders
from src.model.beat import load_fold_teacher
from src.model.student import build_student
from src.train import DatasetConfig, KDTrainConfig, KDLossConfig, KDTrainer
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fresh Low/High CED-Tiny students for the imp/moe pipeline."
    )
    parser.add_argument("--config", type=Path, default=Path("config/kd_ced_tiny.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fold0-checkpoint", type=Path, default=Path("/marimo/pretrained/beats_teacher_fold0_weights.pt"))
    parser.add_argument("--ema-checkpoint", type=Path, default=Path("/marimo/pretrained/beats_teacher_ema_standalone.pt"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--low-epochs", type=int, default=15)
    parser.add_argument("--high-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def train_stage(
    *,
    name: str,
    snr_filter: tuple[float, ...] | None,
    teacher_checkpoint: Path,
    teacher_config: Path,
    raw: dict[str, Any],
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    initialize_from: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    set_seed(seed)
    dataset_config = dataclasses.replace(
        DatasetConfig.from_dict(raw["dataset"]),
        snr_db=snr_filter,
    )
    loaders = build_cached_loaders(
        dataset_config,
        cache_dir,
        batch_size,
        eval_batch_size,
        max_samples=None,
        num_workers=num_workers,
    )
    if len(loaders.class_names) != int(raw["student"]["num_classes"]):
        raise ValueError(
            f"dataset has {len(loaders.class_names)} classes, expected {raw['student']['num_classes']}"
        )

    model = build_student(raw, root=config_path.parent)
    pretrain = model.init_report
    print(
        f"[{name}] CED-Tiny AudioSet initialization: "
        f"loaded={pretrain.get('loaded_tensors', 0)}/148, missing={pretrain.get('missing', 'n/a')}",
        flush=True,
    )
    if int(pretrain.get("loaded_tensors", 0)) < 148:
        raise RuntimeError(f"{name}: incomplete CED-Tiny pretrained load; expected at least 148 tensors")

    if initialize_from is not None:
        report = model.load_pretrained(initialize_from)
        if not report.is_clean:
            raise RuntimeError(
                f"{name}: Low-Student initialization mismatch; missing={report.missing[:5]}, "
                f"unexpected={report.unexpected[:5]}"
            )
        print(f"[{name}] initialized from newly trained Low-Student: {initialize_from}", flush=True)

    teacher, teacher_metadata = load_fold_teacher(teacher_checkpoint, teacher_config, device)
    print(f"[{name}] frozen BEATs teacher loaded: {teacher_metadata}", flush=True)
    base_train = KDTrainConfig.from_dict(raw["train"])
    if name == "Low-Student":
        train_config = dataclasses.replace(
            base_train,
            epochs=epochs,
            patience=max(base_train.patience, epochs),
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            train_eval_clips=0,
            selection_metric="macro_acc",
            gap_penalty=0.0,
        )
    else:
        train_config = dataclasses.replace(
            base_train,
            epochs=epochs,
            patience=epochs,
            lr=5.0e-5,
            weight_decay=0.05,
            warmup_steps=100,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            mixup=0.0,
            train_eval_clips=0,
            selection_metric="top1_accuracy",
            gap_penalty=0.0,
        )
    loss_config = KDLossConfig.from_dict(raw["loss"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"moe_train_{name.lower().replace('-', '_')}", output_dir / "train.log")
    checkpoint_path = output_dir / "last_best_checkpoint.pt"
    state_path = output_dir / "train_state.pt"

    def save_epoch_history(epoch: int, history: list[dict[str, float]]) -> None:
        save_json(output_dir / "history.json", {"name": name, "history": history})

    trainer = KDTrainer(
        model=model,
        teacher=teacher,
        config=train_config,
        loss_config=loss_config,
        device=device,
        logger=logger,
        name=name,
        checkpoint_path=checkpoint_path,
        train_state_path=state_path,
        input_rms=float(raw["teacher"]["input_rms"]),
        train_eval_loader=None,
        on_epoch=save_epoch_history,
        resume=False,
    )
    print(
        f"[{name}] START snr_filter={snr_filter if snr_filter is not None else 'all'} "
        f"train={len(loaders.train.dataset)} validation={len(loaders.validation.dataset)} "
        f"epochs={epochs} selection={train_config.selection_metric}",
        flush=True,
    )
    result = trainer.fit(loaders.train, loaders.validation)
    summary = {
        "name": name,
        "snr_filter": snr_filter,
        "teacher_checkpoint": str(teacher_checkpoint),
        "initialized_from": str(initialize_from) if initialize_from else "mispeech/ced-tiny AudioSet",
        "selection_metric": result.selection_metric,
        "best_score": result.best_score,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "checkpoint": str(checkpoint_path),
        "history": result.history,
    }
    save_json(output_dir / "summary.json", summary)
    print(
        f"[{name}] COMPLETE best_epoch={result.best_epoch} "
        f"best_{result.selection_metric}={result.best_score:.5f} checkpoint={checkpoint_path}",
        flush=True,
    )
    del trainer, teacher, model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path, summary


def update_moe_checkpoint_config(path: Path, low_checkpoint: Path, high_checkpoint: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw.setdefault("checkpoints", {})["low_student"] = str(low_checkpoint)
    raw["checkpoints"]["high_student"] = str(high_checkpoint)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not data_root.joinpath("train", "manifest.csv").is_file():
        raise FileNotFoundError(data_root / "train/manifest.csv")
    if not data_root.joinpath("eval", "manifest.csv").is_file():
        raise FileNotFoundError(data_root / "eval/manifest.csv")
    for split in ("train", "validation"):
        for suffix in ("npy", "done"):
            candidate = cache_dir / f"{split}.{suffix}"
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
    for teacher_checkpoint in (args.fold0_checkpoint, args.ema_checkpoint):
        if not teacher_checkpoint.is_file():
            raise FileNotFoundError(teacher_checkpoint)

    os.environ["MLNC_MIXTURE_36L"] = str(data_root)
    raw = read_config(config_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"TRAIN_FROM_SCRATCH device={device} data={data_root} cache={cache_dir}", flush=True)
    print("Low-Student uses all training SNRs; High-Student fine-tunes the fresh Low checkpoint on SNR=20 dB.", flush=True)
    loss = raw["loss"]
    print(
        f"Loss: label-smoothed CE + {loss['weight']} * T^2 KL(teacher || student), "
        f"T={loss['temperature']}; BEATs teachers are frozen.",
        flush=True,
    )

    low_checkpoint, low_summary = train_stage(
        name="Low-Student",
        snr_filter=None,
        teacher_checkpoint=args.fold0_checkpoint,
        teacher_config=project_root / "config" / "beat.yaml",
        raw=raw,
        config_path=config_path,
        cache_dir=cache_dir,
        output_dir=args.output_root / "low",
        device=device,
        seed=args.seed,
        epochs=args.low_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    high_checkpoint, high_summary = train_stage(
        name="High-Student",
        snr_filter=(20.0,),
        teacher_checkpoint=args.ema_checkpoint,
        teacher_config=project_root / "config" / "beat.yaml",
        raw=raw,
        config_path=config_path,
        cache_dir=cache_dir,
        output_dir=args.output_root / "high",
        device=device,
        seed=args.seed + 1,
        epochs=args.high_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        initialize_from=low_checkpoint,
    )

    update_moe_checkpoint_config(project_root / "config" / "moe.yaml", low_checkpoint, high_checkpoint)
    save_json(args.output_root / "summary.json", {"low": low_summary, "high": high_summary})
    print(f"TRAINING_COMPLETE low={low_checkpoint} high={high_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
