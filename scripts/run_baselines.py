from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune import METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES  # noqa: E402
from scripts.load_dataset import LabeledAudioDataset, build_dataloader, build_dataset, subsample  # noqa: E402
from src.model.baseline import BaselineModel  # noqa: E402
from src.train import BEATTrainConfig, BEATTrainer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.paths import RunPaths  # noqa: E402
from src.utils.plots import plot_confusion_matrix  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

BASELINES_CONFIG = ROOT / "config" / "baselines.yaml"
RESUME_FILENAME = "resume.pt"
RESULT_COLUMNS = (
    "model",
    "family",
    "architecture",
    "params_m",
    "best_epoch",
    "epochs_run",
    "train_seconds",
    *METRIC_NAMES,
)
COMPARISON_COLUMNS = ("rank", "model", "family", "architecture", "params_m", *METRIC_NAMES)
COMPARISON_FILENAME = "comparison.csv"
SORT_METRIC = "macro_f1"


def read_baselines_config(path: str | Path = BASELINES_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def baseline_names(path: str | Path = BASELINES_CONFIG) -> list[str]:
    return list(read_baselines_config(path)["models"])


def count_millions(model: torch.nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def available_frames(duration: float, frontend: dict[str, Any]) -> int:
    samples = int(duration * frontend["sample_rate"])
    padded = samples + max(frontend["n_fft"] - frontend["win_length"], 0)
    return (padded - frontend["n_fft"]) // frontend["hop_length"] + 1


def check_frames(duration: float, frontend: dict[str, Any], num_frames: int) -> None:
    frames = available_frames(duration, frontend)
    if num_frames > frames:
        raise ValueError(f"num_frames {num_frames} exceeds the {frames} frames of a {duration}s clip")


def describe_baselines(path: str | Path = BASELINES_CONFIG, num_classes: int | None = None) -> list[dict[str, Any]]:
    raw = read_baselines_config(path)
    rows = []
    for name in raw["models"]:
        model = BaselineModel.from_yaml(path, name)
        if num_classes is not None:
            model = BaselineModel(replace(model.config, num_classes=num_classes), raw.get("frontend"))
        rows.append(
            {
                "model": name,
                "family": model.config.family,
                "architecture": model.config.label or model.config.arch,
                "arch": model.config.arch,
                "params_m": round(count_millions(model), 2),
            }
        )
    return rows


def build_loaders(
    dataset_name: str,
    duration: float,
    batch_size: int,
    eval_batch_size: int,
    *,
    max_samples: int | None = None,
    num_workers: int = 4,
) -> dict[str, Any]:
    train = build_dataset(dataset_name, "train", duration=duration)
    validation = build_dataset(dataset_name, "validation", duration=duration)
    if not isinstance(train, LabeledAudioDataset):
        raise ValueError(f"dataset '{dataset_name}' has no class labels")
    return {
        "train": build_dataloader(subsample(train, max_samples), batch_size, True, num_workers),
        "validation": build_dataloader(subsample(validation, max_samples), eval_batch_size, False, num_workers),
        "class_names": train.class_names,
    }


def build_baseline(raw: dict[str, Any], config_path: str | Path, name: str, num_classes: int) -> BaselineModel:
    model = BaselineModel.from_yaml(config_path, name)
    return BaselineModel(replace(model.config, num_classes=num_classes), raw["frontend"])


def write_result_csv(rows: list[dict[str, Any]], path: str | Path, columns: tuple[str, ...] = RESULT_COLUMNS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def best_epoch_scores(history: list[dict[str, Any]], epoch: int) -> dict[str, float]:
    for entry in history:
        if entry["epoch"] == epoch:
            return {key.removeprefix("val_"): value for key, value in entry.items() if key.startswith("val_")}
    raise ValueError(f"epoch {epoch} is missing from the validation history")


def train_baseline(
    name: str,
    model: BaselineModel,
    loaders: dict[str, DataLoader],
    config: BEATTrainConfig,
    run: RunPaths,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    history_path = run.history_path(name)
    if history_path.is_file():
        logger.info("%s: finished earlier, reading %s", name, history_path)
        return json.loads(history_path.read_text(encoding="utf-8"))["row"]

    checkpoint_path = run.checkpoint_path(name)
    resume_path = checkpoint_path.with_name(RESUME_FILENAME)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    params_m = count_millions(model)
    logger.info("%s: %s | %.2fM params", name, model.config.arch, params_m)

    trainer = BEATTrainer(model, config, device, logger, name, checkpoint_path, resume_path=resume_path)
    if resume_path.is_file():
        trainer.load_resume(resume_path)
    started = time.time()
    fit = trainer.fit(loaders["train"], loaders["validation"])
    confusion = trainer.confusion_matrix(loaders["validation"], desc=f"{name} - confusion")

    scores = best_epoch_scores(trainer.history, fit.best_epoch)
    row = {
        "model": name,
        "family": model.config.family,
        "architecture": model.config.label or model.config.arch,
        "params_m": round(params_m, 2),
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        "train_seconds": round(time.time() - started, 1),
        **{metric: scores[metric] for metric in METRIC_NAMES},
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"row": row, "history": trainer.history, "confusion_matrix": confusion.tolist()}
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("%s: validation | %s", name, " ".join(f"{metric} {row[metric]:.4f}" for metric in METRIC_NAMES))
    return row


def run_baselines(
    run: RunPaths,
    device: torch.device,
    logger: logging.Logger,
    *,
    names: list[str] | None = None,
    config_path: str | Path = BASELINES_CONFIG,
    max_samples: int | None = None,
    num_workers: int = 4,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    raw = read_baselines_config(config_path)
    selected = names or list(raw["models"])
    unknown = [name for name in selected if name not in raw["models"]]
    if unknown:
        raise KeyError(f"unknown baselines {unknown}, expected some of {sorted(raw['models'])}")

    dataset = raw["dataset"]
    check_frames(dataset["duration"], raw["frontend"], raw["model"]["num_frames"])
    base_config = BEATTrainConfig.from_dict(raw["train"])
    seed = int(raw["train"].get("seed", 42))
    if smoke:
        base_config = replace(base_config, epochs=1, patience=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
        max_samples, num_workers = SMOKE_SAMPLES, 0

    run.create()
    rows: list[dict[str, Any]] = []
    for name in selected:
        batch_size = SMOKE_BATCH if smoke else raw["models"][name].get("batch_size", base_config.batch_size)
        config = replace(base_config, batch_size=batch_size)
        loaders = build_loaders(
            dataset["name"],
            dataset["duration"],
            config.batch_size,
            config.eval_batch_size,
            max_samples=max_samples,
            num_workers=num_workers,
        )
        set_seed(seed)
        model = build_baseline(raw, config_path, name, len(loaders["class_names"]))
        rows.append(train_baseline(name, model, loaders, config, run, device, logger))
        write_result_csv(rows, run.result_file)
        del model, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info("results written to %s", run.result_file)
    return rows


def evaluate_baselines(
    run: RunPaths,
    device: torch.device,
    logger: logging.Logger,
    *,
    split: str = "validation",
    names: list[str] | None = None,
    config_path: str | Path = BASELINES_CONFIG,
    max_samples: int | None = None,
    num_workers: int = 4,
) -> dict[str, dict[str, Any]]:
    raw = read_baselines_config(config_path)
    selected = names or [name for name in raw["models"] if run.checkpoint_path(name).is_file()]
    missing = [name for name in selected if not run.checkpoint_path(name).is_file()]
    if missing:
        raise FileNotFoundError(f"no checkpoint under {run.checkpoint_dir} for {missing}")

    dataset = raw["dataset"]
    data = build_dataset(dataset["name"], split, duration=dataset["duration"])
    if len(data) == 0:
        raise ValueError(f"split '{split}' of {dataset['name']} has no samples")
    config = BEATTrainConfig.from_dict(raw["train"])
    loader = build_dataloader(subsample(data, max_samples), config.eval_batch_size, False, num_workers)

    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        model = build_baseline(raw, config_path, name, data.num_classes)
        load_checkpoint(run.checkpoint_path(name), model)
        model.to(device)
        trainer = BEATTrainer(model, config, device, logger, name, run.checkpoint_path(name))
        scores = trainer.evaluate(loader, desc=f"{name} - {split}")
        confusion = trainer.confusion_matrix(loader, desc=f"{name} - confusion")
        row = {
            "model": name,
            "family": model.config.family,
            "architecture": model.config.label or model.config.arch,
            "params_m": round(count_millions(model), 2),
            **{metric: scores[metric] for metric in METRIC_NAMES},
        }
        results[name] = {"row": row, "confusion_matrix": confusion}
        logger.info("%s: %s | %s", name, split, " ".join(f"{metric} {row[metric]:.4f}" for metric in METRIC_NAMES))
        del model, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def load_results(run: RunPaths) -> list[dict[str, Any]]:
    paths = sorted(run.history_dir.glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8"))["row"] for path in paths]


def load_confusions(run: RunPaths) -> dict[str, np.ndarray]:
    payloads = {
        path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(run.history_dir.glob("*.json"))
    }
    return {name: np.asarray(payload["confusion_matrix"], dtype=np.int64) for name, payload in payloads.items()}


def comparison_table(rows: list[dict[str, Any]], sort_by: str = SORT_METRIC) -> list[dict[str, Any]]:
    if rows and sort_by not in rows[0]:
        raise KeyError(f"cannot sort by '{sort_by}', expected one of {sorted(rows[0])}")
    ordered = sorted(rows, key=lambda row: row[sort_by], reverse=True)
    return [{"rank": rank, **row} for rank, row in enumerate(ordered, start=1)]


def best_models(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {metric: max(rows, key=lambda row: row[metric])["model"] for metric in METRIC_NAMES}


def write_comparison(rows: list[dict[str, Any]], run: RunPaths, sort_by: str = SORT_METRIC) -> Path:
    table = comparison_table(rows, sort_by)
    return write_result_csv(table, run.result_dir / COMPARISON_FILENAME, COMPARISON_COLUMNS)


def plot_baseline_confusions(
    matrices: dict[str, np.ndarray], class_names: list[str], run: RunPaths, split: str = "validation"
) -> dict[str, Path]:
    return {
        name: plot_confusion_matrix(
            matrix, class_names, run.figure_dir / f"confusion_{split}_{name}.png", f"{name} | {split}"
        )
        for name, matrix in matrices.items()
    }


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...], digits: int = 4) -> str:
    def cell(value: Any) -> str:
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def write_variant_config(
    target: str | Path,
    *,
    dataset_name: str,
    duration: float,
    num_frames: int,
    source: str | Path = BASELINES_CONFIG,
) -> Path:
    raw = read_baselines_config(source)
    raw["dataset"] = {"name": dataset_name, "duration": duration}
    raw["frontend"]["num_frames"] = num_frames
    raw["model"]["num_frames"] = num_frames
    check_frames(duration, raw["frontend"], num_frames)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
    return target
