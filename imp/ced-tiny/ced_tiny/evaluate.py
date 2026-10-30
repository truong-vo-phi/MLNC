from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import ClassificationMetrics
from src.train.reviewkd_trainer import snr_scores

from .data import read_ytids_from_file
from .model import rms_normalize
from .runtime import autocast_context

METRIC_NAMES = (
    "top1_accuracy",
    "top3_accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "micro_f1",
    "mAP",
    "macro_auc",
)


@dataclass
class EvaluationPredictions:
    probs: np.ndarray
    labels: np.ndarray
    snrs: np.ndarray
    sample_ids: list[str]
    noise_ytids: list[str]

    def __len__(self) -> int:
        return len(self.labels)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    separator: nn.Module | None = None,
    target_rms: float = 0.07925,
    amp_mode: str = "none",
    desc: str = "evaluating",
) -> EvaluationPredictions:
    model.eval()
    if separator is not None:
        separator.eval()

    all_probs: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_snrs: list[Tensor] = []
    all_sample_ids: list[str] = []
    all_noise_ytids: list[str] = []

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        waveform = batch["waveform"].to(device)
        labels = batch["target"].argmax(dim=1).cpu()
        snr = batch["snr_db"].cpu()
        sample_id = batch["sample_id"]
        noise_ytid = batch.get("noise_ytid", [""] * len(labels))

        if separator is not None:
            waveform = separator(waveform)
        if target_rms > 0:
            waveform = rms_normalize(waveform, target_rms)

        with autocast_context(device, amp_mode):
            probs = model.predict_proba(waveform).float().cpu()

        all_probs.append(probs)
        all_labels.append(labels)
        all_snrs.append(snr)
        all_sample_ids.extend(sample_id if isinstance(sample_id, list) else list(sample_id))
        all_noise_ytids.extend(noise_ytid if isinstance(noise_ytid, list) else list(noise_ytid))

    return EvaluationPredictions(
        probs=torch.cat(all_probs, dim=0).numpy(),
        labels=torch.cat(all_labels, dim=0).numpy(),
        snrs=torch.cat(all_snrs, dim=0).numpy(),
        sample_ids=all_sample_ids,
        noise_ytids=all_noise_ytids,
    )


def compute_metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int = 36) -> dict[str, float]:
    cm = ClassificationMetrics(num_classes=num_classes)
    cm.update(torch.from_numpy(labels), torch.from_numpy(probs))
    return cm.compute().to_dict()


def compute_snr_breakdown(
    predictions: EvaluationPredictions,
    snr_levels: Sequence[float],
    num_classes: int = 36,
) -> dict[str, dict[str, float]]:
    return snr_scores(predictions.probs, predictions.labels, predictions.snrs, num_classes)


def compute_unseen_metrics(
    predictions: EvaluationPredictions,
    eval_segments_path: str | Path | None,
    train_segments_paths: Sequence[str | Path],
    snr_levels: Sequence[float],
    num_classes: int = 36,
) -> tuple[dict[str, float] | None, dict[str, dict[str, float]] | None, int]:
    if not eval_segments_path:
        return None, None, 0
    eval_ytids = read_ytids_from_file(eval_segments_path)
    if not eval_ytids:
        return None, None, 0

    seen_ytids: set[str] = set()
    for path in train_segments_paths:
        seen_ytids |= read_ytids_from_file(path)

    mask = [
        predictions.noise_ytids[i] in eval_ytids and predictions.noise_ytids[i] not in seen_ytids
        for i in range(len(predictions))
    ]
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None, None, 0

    sub_probs = predictions.probs[indices]
    sub_labels = predictions.labels[indices]
    sub_snrs = predictions.snrs[indices]

    overall = compute_metrics(sub_probs, sub_labels, num_classes)
    by_snr = snr_scores(sub_probs, sub_labels, sub_snrs, num_classes)
    return overall, by_snr, len(indices)


def write_result_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_snr_breakdown_csv(path: Path, scores: dict[str, float] | dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if any(isinstance(v, dict) for v in scores.values()):
        for snr_key, row in scores.items():
            if isinstance(row, dict):
                rows.append({"snr_db": snr_key, **row})
    else:
        levels: set[int] = set()
        for k in scores:
            if k.startswith("acc_snr_"):
                levels.add(int(float(k.replace("acc_snr_", ""))))
        for lvl in sorted(levels):
            row = {"snr_db": lvl}
            for name in METRIC_NAMES:
                key = f"{name}_snr_{lvl}"
                if key in scores:
                    row[name] = scores[key]
            rows.append(row)
    if rows:
        write_result_csv(path, rows)
