from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import (  # noqa: E402
    SyntheticDataset,
    build_cached_loader,
    build_cached_loaders,
    build_dataloader,
)
from src.metrics import NINE_METRICS  # noqa: E402
from src.model.beat import BEATClassifier, load_fold_teacher  # noqa: E402
from src.model.student import (  # noqa: E402
    CEDStudent,
    LogMel,
    LogMelConfig,
    SpecAugment,
    SpecAugmentConfig,
    StudentConfig,
)
from src.train import DatasetConfig, KDLossConfig, KDTrainConfig, KDTrainer  # noqa: E402
from src.utils.files import mirror_file  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.paths import RunPaths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

CONFIG = ROOT / "config" / "kd_ced_tiny.yaml"
SNR_LEVELS = (-5, 0, 5, 10, 15, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEATs -> CED-Tiny distillation on Mixture-36l")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="distill CED-Tiny from a frozen BEATs teacher")
    train.add_argument("--config", type=Path, default=CONFIG)
    train.add_argument("--noise-cache", type=Path, default=None)
    train.add_argument("--output-dir", type=Path, default=ROOT / "output")
    train.add_argument("--run-id", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--ced-weights", type=Path, default=None)
    train.add_argument("--backup-dir", type=Path, default=None)

    evaluate = subparsers.add_parser("eval", help="evaluate a student checkpoint")
    evaluate.add_argument("--config", type=Path, default=CONFIG)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--sample-ids-file", type=Path, default=None)
    evaluate.add_argument("--noise-cache", type=Path, default=None)
    evaluate.add_argument("--output-dir", type=Path, default=ROOT / "output")
    evaluate.add_argument("--run-id", default=None)
    evaluate.add_argument("--max-samples", type=int, default=None)
    evaluate.add_argument("--num-workers", type=int, default=None)
    evaluate.add_argument("--device", default=None)

    ensemble_parser = subparsers.add_parser("ensemble", help="evaluate student + teacher ensemble")
    ensemble_parser.add_argument("--config", type=Path, default=CONFIG)
    ensemble_parser.add_argument("--checkpoint", type=Path, required=True)
    ensemble_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    ensemble_parser.add_argument("--weight", type=float, default=0.4, help="weight for student (1-weight for teacher)")
    ensemble_parser.add_argument("--sample-ids-file", type=Path, default=None)
    ensemble_parser.add_argument("--noise-cache", type=Path, default=None)
    ensemble_parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ensemble_parser.add_argument("--run-id", default=None)
    ensemble_parser.add_argument("--max-samples", type=int, default=None)
    ensemble_parser.add_argument("--num-workers", type=int, default=None)
    ensemble_parser.add_argument("--device", default=None)

    params = subparsers.add_parser("params", help="print the student parameter table")
    params.add_argument("--config", type=Path, default=CONFIG)
    params.add_argument("--ced-weights", type=Path, default=None)
    params.add_argument("--with-init", action="store_true")

    smoke = subparsers.add_parser("smoke", help="run one epoch on synthetic data")
    smoke.add_argument("--config", type=Path, default=CONFIG)
    smoke.add_argument("--samples", type=int, default=8)
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--output-dir", type=Path, default=ROOT / "output" / "test")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def select_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_student(raw: dict[str, Any], overrides: dict[str, Any] | None = None) -> CEDStudent:
    values = {**raw["student"], **(overrides or {})}
    ced_path = values.get("ced_path")
    if ced_path and not Path(ced_path).is_absolute():
        values["ced_path"] = str(resolve_path(ced_path))
    student = StudentConfig.from_dict(values)
    frontend = LogMelConfig.from_dict(raw["frontend"])
    spec_augment = SpecAugmentConfig.from_dict(raw["specaugment"])
    return CEDStudent(student, LogMel(**frontend.to_dict()), SpecAugment(**spec_augment.to_dict()))


def build_run(args: argparse.Namespace, experiment: str) -> RunPaths:
    if args.run_id:
        return RunPaths.from_run_id(args.run_id, args.output_dir)
    return RunPaths.new(experiment, args.output_dir)


def save_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def result_rows(metrics: dict[str, float], split: str) -> list[dict[str, Any]]:
    def row(subset: str, values: dict[str, float]) -> dict[str, Any]:
        record = {"split": split, "snr_db": subset}
        for name in NINE_METRICS:
            record[name] = values.get(name)
        for name in ("macro_acc", "mAP_macro", "acc", "score", "gap"):
            record[name] = values.get(name)
        return record

    rows = [row("overall", metrics)]
    for snr in SNR_LEVELS:
        values = {name: metrics.get(f"{name}_snr_{snr}") for name in NINE_METRICS}
        values["score"] = metrics.get("score")
        rows.append(row(str(snr), values))
    return rows


def write_result_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "snr_db", *NINE_METRICS, "macro_acc", "mAP_macro", "acc", "score", "gap"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolved_cache(raw: dict[str, Any], override: Path | None) -> Path:
    value = override or raw["dataset"].get("noise_cache")
    if value is None:
        raise ValueError("a noise cache is required; pass --noise-cache or set dataset.noise_cache")
    path = resolve_path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"noise cache directory does not exist: {path}")
    return path


def train(args: argparse.Namespace) -> None:
    raw = read_config(args.config)
    device = select_device(args.device)
    run = build_run(args, "kd-ced-tiny").create()
    logger = setup_logger("kd-v5", run.log_file)
    dataset_config = DatasetConfig.from_dict(raw["dataset"])
    train_config = KDTrainConfig.from_dict(raw["train"])
    loss_config = KDLossConfig.from_dict(raw["loss"])
    if args.epochs is not None:
        train_config = dataclasses.replace(train_config, epochs=args.epochs)
    if args.num_workers is not None:
        train_config = dataclasses.replace(train_config, num_workers=args.num_workers)
    if args.seed is not None:
        train_config = dataclasses.replace(train_config, seed=args.seed)
    set_seed(train_config.seed)
    cache = resolved_cache(raw, args.noise_cache)
    logger.info("run %s | device %s | cache %s", run.run_id, device, cache)
    loaders = build_cached_loaders(
        dataset_config,
        cache,
        train_config.batch_size,
        train_config.eval_batch_size,
        args.max_samples,
        train_config.num_workers,
    )
    train_eval = (
        build_cached_loader(
            dataset_config,
            "train",
            cache,
            train_config.eval_batch_size,
            train_config.train_eval_clips,
            train_config.num_workers,
            shuffle=False,
        )
        if train_config.train_eval_clips > 0
        else None
    )

    overrides: dict[str, Any] = {}
    if args.ced_weights is not None:
        overrides["ced_path"] = str(args.ced_weights)
    student = build_student(raw, overrides)
    if len(loaders.class_names) != student.config.num_classes:
        raise ValueError(
            f"dataset has {len(loaders.class_names)} classes, student expects {student.config.num_classes}"
        )
    teacher, metadata = load_fold_teacher(
        resolve_path(raw["teacher"]["checkpoint"]), resolve_path(raw["teacher"]["config"]), device
    )
    logger.info("teacher %s | student %s", metadata, student.init_report)
    logger.info("student parameters %.2fM", student.num_parameters() / 1e6)

    backup_dir = args.backup_dir
    on_checkpoint = (
        (lambda path, epoch, score: mirror_file(path, backup_dir, run.base_dir)) if backup_dir is not None else None
    )

    def on_epoch(epoch: int, history: list[dict[str, float]]) -> None:
        save_history(
            run.history_path("student"),
            {"run_id": run.run_id, "epoch": epoch, "config": raw, "history": history},
        )
        if backup_dir is not None:
            mirror_file(run.history_path("student"), backup_dir, run.base_dir)

    trainer = KDTrainer(
        student,
        teacher,
        train_config,
        loss_config,
        device,
        logger,
        "kd-ced-tiny",
        run.checkpoint_path("student"),
        run.train_state_path("student"),
        float(raw["teacher"]["input_rms"]),
        train_eval_loader=train_eval,
        on_checkpoint=on_checkpoint,
        on_epoch=on_epoch,
        resume=args.run_id is not None,
    )
    result = trainer.fit(loaders.train, loaders.validation)
    metrics = trainer.evaluate(loaders.validation, desc="best-aware validation")
    write_result_csv(run.result_file, result_rows(metrics, "validation"))
    save_history(
        run.history_path("student"),
        {
            "run_id": run.run_id,
            "config": raw,
            "result": result.to_dict(),
            "validation": metrics,
        },
    )
    if backup_dir is not None:
        mirror_file(run.result_file, backup_dir, run.base_dir)
        mirror_file(run.history_path("student"), backup_dir, run.base_dir)
    logger.info(
        "done %s | best epoch %d | macro_acc %.4f | mAP %.4f | score %.4f",
        run.run_id,
        result.best_epoch,
        metrics["macro_acc"],
        metrics["mAP"],
        metrics["score"],
    )


def evaluate(args: argparse.Namespace) -> None:
    raw = read_config(args.config)
    device = select_device(args.device)
    run = build_run(args, f"kd-ced-tiny-eval-{args.split}").create()
    logger = setup_logger("kd-v5", run.log_file)
    dataset_config = DatasetConfig.from_dict(raw["dataset"])
    train_config = KDTrainConfig.from_dict(raw["train"])
    loss_config = KDLossConfig.from_dict(raw["loss"])
    if args.num_workers is not None:
        train_config = dataclasses.replace(train_config, num_workers=args.num_workers)
    cache = resolved_cache(raw, args.noise_cache)
    student = build_student(raw)
    student.load_pretrained(args.checkpoint)
    student.to(device)
    loader = build_cached_loader(
        dataset_config,
        args.split,
        cache,
        train_config.eval_batch_size,
        args.max_samples,
        train_config.num_workers,
        shuffle=False,
        sample_ids_file=args.sample_ids_file,
    )
    trainer = KDTrainer(
        student,
        None,
        train_config,
        loss_config,
        device,
        logger,
        "kd-ced-tiny",
        run.checkpoint_path("student"),
        run.train_state_path("student"),
        float(raw["teacher"]["input_rms"]),
    )
    metrics = trainer.evaluate(loader, desc=args.split)
    write_result_csv(run.result_file, result_rows(metrics, args.split))
    save_history(
        run.history_path("student"),
        {"run_id": run.run_id, "checkpoint": str(args.checkpoint), "split": args.split, "metrics": metrics},
    )
    logger.info(
        "%s | %s | macro_acc %.4f | mAP %.4f",
        run.run_id,
        args.split,
        metrics["macro_acc"],
        metrics["mAP"],
    )


def ensemble(args: argparse.Namespace) -> None:
    raw = read_config(args.config)
    device = select_device(args.device)
    run = build_run(args, f"kd-ced-tiny-ensemble-{args.split}").create()
    logger = setup_logger("kd-v5", run.log_file)
    dataset_config = DatasetConfig.from_dict(raw["dataset"])
    train_config = KDTrainConfig.from_dict(raw["train"])
    cache = resolved_cache(raw, args.noise_cache)
    student = build_student(raw)
    student.load_pretrained(args.checkpoint)
    student.to(device).eval()
    teacher, _ = load_fold_teacher(
        resolve_path(raw["teacher"]["checkpoint"]), resolve_path(raw["teacher"]["config"]), device
    )
    teacher.eval()
    loader = build_cached_loader(
        dataset_config,
        args.split,
        cache,
        train_config.eval_batch_size,
        args.max_samples,
        train_config.num_workers if args.num_workers is None else args.num_workers,
        shuffle=False,
        sample_ids_file=args.sample_ids_file,
    )
    input_rms = float(raw["teacher"]["input_rms"])
    s_probs, t_probs, labels, snrs = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ensemble {args.split}"):
            wave = batch["waveform"].to(device, non_blocking=True)
            noise = rms_normalize(wave, input_rms)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=train_config.amp and device.type == "cuda"):
                s_logits = student(noise).logits
                t_logits = teacher(noise).logits
            s_probs.append(s_logits.float().softmax(dim=-1).cpu().numpy())
            t_probs.append(t_logits.float().softmax(dim=-1).cpu().numpy())
            labels.append(batch["target"].argmax(dim=-1).numpy())
            snrs.append(batch["snr_db"].numpy())
    s_probs = np.concatenate(s_probs, axis=0)
    t_probs = np.concatenate(t_probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    snrs = np.concatenate(snrs, axis=0)
    blended = args.weight * s_probs + (1.0 - args.weight) * t_probs
    metrics = snr_scores(blended, labels, snrs, student.config.num_classes)
    write_result_csv(run.result_file, result_rows(metrics, args.split))
    save_history(
        run.history_path("student"),
        {"run_id": run.run_id, "checkpoint": str(args.checkpoint), "split": args.split, "weight": args.weight, "metrics": metrics},
    )
    logger.info(
        "ENSEMBLE %s | %s | weight %.2f | top1 %.4f | macro_acc %.4f | mAP %.4f | mAP_macro %.4f",
        run.run_id,
        args.split,
        args.weight,
        metrics["top1_accuracy"],
        metrics["macro_acc"],
        metrics["mAP"],
        metrics["mAP_macro"],
    )


def params(args: argparse.Namespace) -> None:
    raw = read_config(args.config)
    overrides: dict[str, Any] = {} if args.with_init else {"init": "none"}
    if args.ced_weights is not None:
        overrides["ced_path"] = str(args.ced_weights)
    student = build_student(raw, overrides)
    groups = student.describe_params()
    total = sum(groups.values())
    if args.with_init:
        print(f"init {student.config.init} | {student.init_report}")
    print("PARAMS student CED-Tiny")
    for name, count in groups.items():
        print(f"PARAMS  {name:36s} {count:>12,d}  {100 * count / total:5.1f}%")
    print(f"PARAMS  {'TOTAL':36s} {total:>12,d}  ({total / 1e6:.2f}M)")


def smoke(args: argparse.Namespace) -> None:
    raw = read_config(args.config)
    device = select_device(args.device)
    run = RunPaths.new("smoke", args.output_dir).create()
    logger = setup_logger("kd-v5", run.log_file)
    train_config = dataclasses.replace(
        KDTrainConfig.from_dict(raw["train"]),
        epochs=1,
        patience=2,
        batch_size=2,
        eval_batch_size=2,
        warmup_steps=1,
        train_eval_clips=4,
        num_workers=0,
        amp=False,
    )
    loss_config = KDLossConfig.from_dict(raw["loss"])
    dataset = SyntheticDataset(length=args.samples, num_classes=36)
    train_loader = build_dataloader(dataset, 2, shuffle=True, num_workers=0)
    val_loader = build_dataloader(dataset, 2, shuffle=False, num_workers=0)
    student = build_student(raw, {"init": "none"})
    teacher = BEATClassifier.from_yaml(resolve_path(raw["teacher"]["config"])).to(device).freeze()
    trainer = KDTrainer(
        student,
        teacher,
        train_config,
        loss_config,
        device,
        logger,
        "smoke",
        run.checkpoint_path("student"),
        run.train_state_path("student"),
        float(raw["teacher"]["input_rms"]),
        train_eval_loader=val_loader,
    )
    result = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(val_loader, desc="smoke validation")
    write_result_csv(run.result_file, result_rows(metrics, "synthetic"))
    save_history(
        run.history_path("student"),
        {"run_id": run.run_id, "result": result.to_dict(), "metrics": metrics, "config": raw},
    )
    logger.info(
        "smoke done %s | best epoch %d | loss finite | macro_acc %.4f",
        run.run_id,
        result.best_epoch,
        metrics["macro_acc"],
    )


def main() -> None:
    args = parse_args()
    {"train": train, "eval": evaluate, "ensemble": ensemble, "params": params, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
