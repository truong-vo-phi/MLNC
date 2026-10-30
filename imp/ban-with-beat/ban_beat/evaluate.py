from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import ClassificationMetrics
from src.utils.plots import BLUE, GRID, ORANGE, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

from .data import dataset_rows
from .runtime import autocast_context, softmax_numpy

HIGH_SNR = 15.0
CURVE_METRICS = ("top1_accuracy", "macro_f1", "mAP")
PALETTE = (BLUE, ORANGE, "#7a5af8", "#12a594", "#b54708", "#667085")


@dataclass(frozen=True)
class Predictions:
    sample_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    snr_db: np.ndarray

    def __post_init__(self) -> None:
        if self.labels.shape[0] != self.probabilities.shape[0] or self.labels.shape[0] != self.snr_db.shape[0]:
            raise ValueError("labels, probabilities and snr_db must have the same number of rows")
        if self.labels.shape[0] != len(self.sample_ids):
            raise ValueError("sample_ids and prediction rows differ")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must be a 2D array")


@torch.no_grad()
def collect_predictions(
    predict: Callable[[Tensor], Tensor],
    loader: DataLoader,
    device: torch.device,
    amp: str,
    desc: str = "evaluate",
) -> Predictions:
    sample_ids: list[str] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    snrs: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        waveform = batch["waveform"].to(device)
        with autocast_context(device, amp):
            logits = predict(waveform)
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.append(batch["target"].argmax(dim=1).cpu().numpy())
        snrs.append(batch["snr_db"].cpu().numpy())
        sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
    if not probabilities:
        raise ValueError("loader produced no batches")
    return Predictions(
        sample_ids=tuple(sample_ids),
        labels=np.concatenate(labels).astype(np.int64),
        probabilities=np.concatenate(probabilities).astype(np.float32),
        snr_db=np.concatenate(snrs).astype(np.float32),
    )


def predictions_from_cache(dataset: Any, cache: Mapping[str, Tensor]) -> Predictions:
    rows = dataset_rows(dataset)
    sample_ids = tuple(row["sample_id"] for row in rows)
    missing = [sample_id for sample_id in sample_ids if sample_id not in cache]
    if missing:
        raise KeyError(f"teacher cache is missing {len(missing)} samples, first: {missing[0]}")
    logits = np.stack(
        [
            np.asarray(cache[sample_id].detach().cpu(), dtype=np.float32)
            if hasattr(cache[sample_id], "detach")
            else np.asarray(cache[sample_id], dtype=np.float32)
            for sample_id in sample_ids
        ]
    )
    labels = np.array([int(np.argmax(json.loads(row["multi_hot_36"]))) for row in rows], dtype=np.int64)
    snr_db = np.array([float(row["target_snr_db"]) for row in rows], dtype=np.float32)
    return Predictions(sample_ids, labels, softmax_numpy(logits).astype(np.float32), snr_db)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = ClassificationMetrics(num_classes=probabilities.shape[1])
    metrics.update(labels, probabilities)
    return metrics.compute().to_dict()


def metrics_by_snr(
    predictions: Predictions,
    snr_levels: Sequence[float],
    high_snr: float = HIGH_SNR,
) -> dict[str, float]:
    scores = _metrics(predictions.labels, predictions.probabilities)
    scores["acc"] = scores["top1_accuracy"]
    per_top1: list[float] = []
    per_f1: list[float] = []
    per_map: list[float] = []
    high_top1: list[float] = []
    high_f1: list[float] = []
    for level in snr_levels:
        mask = np.isclose(predictions.snr_db, float(level))
        if not mask.any():
            continue
        result = _metrics(predictions.labels[mask], predictions.probabilities[mask])
        suffix = f"{level:g}"
        scores[f"top1_accuracy_snr_{suffix}"] = result["top1_accuracy"]
        scores[f"macro_f1_snr_{suffix}"] = result["macro_f1"]
        scores[f"mAP_snr_{suffix}"] = result["mAP"]
        per_top1.append(result["top1_accuracy"])
        per_f1.append(result["macro_f1"])
        per_map.append(result["mAP"])
        if float(level) >= high_snr:
            high_top1.append(result["top1_accuracy"])
            high_f1.append(result["macro_f1"])
    if not per_top1:
        raise ValueError(f"no predictions match snr_levels {tuple(snr_levels)}")
    scores["top1_accuracy_snr"] = float(np.mean(per_top1))
    scores["macro_f1_snr"] = float(np.mean(per_f1))
    scores["mAP_snr"] = float(np.mean(per_map))
    scores["top1_accuracy_hi"] = float(np.mean(high_top1)) if high_top1 else float("nan")
    scores["macro_f1_hi"] = float(np.mean(high_f1)) if high_f1 else float("nan")
    return scores


EVAL_METRICS = (
    "top1_accuracy",
    "top3_accuracy",
    "mAP",
    "macro_auc",
    "macro_f1",
    "micro_f1",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
)


def metrics_table(predictions: Predictions, snr_levels: Sequence[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in snr_levels:
        mask = np.isclose(predictions.snr_db, float(level))
        if not mask.any():
            continue
        result = _metrics(predictions.labels[mask], predictions.probabilities[mask])
        rows.append({"snr": f"{level:g}", **{name: result[name] for name in EVAL_METRICS}})
    overall = _metrics(predictions.labels, predictions.probabilities)
    rows.append({"snr": "all", **{name: overall[name] for name in EVAL_METRICS}})
    return rows


def snr_curve(predictions: Predictions, snr_levels: Sequence[float]) -> dict[str, dict[float, float]]:
    curves: dict[str, dict[float, float]] = {metric: {} for metric in CURVE_METRICS}
    for level in snr_levels:
        mask = np.isclose(predictions.snr_db, float(level))
        if not mask.any():
            continue
        result = _metrics(predictions.labels[mask], predictions.probabilities[mask])
        for metric in CURVE_METRICS:
            curves[metric][float(level)] = result[metric]
    return curves


def plot_snr_comparison(
    predictions: Mapping[str, Predictions],
    snr_levels: Sequence[float],
    path: str | Path,
    metric: str = "top1_accuracy",
    title: str | None = None,
) -> Path:
    if metric not in CURVE_METRICS:
        raise ValueError(f"metric must be one of {CURVE_METRICS}, got {metric!r}")
    figure, axis = plt.subplots(figsize=(6.6, 3.9), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for position, (name, member) in enumerate(predictions.items()):
        curve = snr_curve(member, snr_levels)[metric]
        levels = sorted(curve)
        axis.plot(
            levels,
            [curve[level] for level in levels],
            color=PALETTE[position % len(PALETTE)],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=name,
        )
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.set_xlabel("SNR (dB)", color=TEXT_SECONDARY)
    axis.set_ylabel(metric, color=TEXT_SECONDARY)
    axis.set_ylim(0.0, 1.02)
    axis.set_title(title or f"{metric} by SNR", fontsize=10, loc="left", color=TEXT_PRIMARY)
    axis.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8)
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return output


def save_predictions(path: str | Path, predictions: Predictions) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(predictions.sample_ids),
        labels=predictions.labels.astype(np.int64),
        probabilities=predictions.probabilities.astype(np.float32),
        snr_db=predictions.snr_db.astype(np.float32),
    )
    return output


def load_predictions(path: str | Path) -> Predictions:
    with np.load(path) as data:
        return Predictions(
            sample_ids=tuple(str(item) for item in data["sample_ids"].tolist()),
            labels=data["labels"].astype(np.int64),
            probabilities=data["probabilities"].astype(np.float32),
            snr_db=data["snr_db"].astype(np.float32),
        )


def write_result_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output
