"""
Comprehensive Metrics Computation for Dual-CED-Tiny:
Computes acc, macro precision, macro recall, macro_f1, micro_f1, mAP,
and provides detailed breakdowns across 6 SNR levels (-5, 0, 5, 10, 15, 20 dB).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import Tensor

SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def compute_metrics(
    predictions: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    num_classes: int = 36,
) -> dict[str, float]:
    """
    Computes 6 core evaluation metrics:
      1. acc: Top-1 Accuracy
      2. precision: Macro Precision
      3. recall: Macro Recall
      4. macro_f1: Macro F1-Score
      5. micro_f1: Micro F1-Score
      6. mAP: Mean Average Precision from softmax probability distributions
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, Tensor):
        targets = targets.detach().cpu().numpy()

    # Probability conversion if logits provided
    if predictions.ndim == 2:
        if (predictions < 0).any() or (predictions > 1).any():
            # Apply softmax to raw logits
            exp_p = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
            probs = exp_p / np.sum(exp_p, axis=-1, keepdims=True)
        else:
            probs = predictions
        pred_labels = np.argmax(probs, axis=-1)
    else:
        pred_labels = predictions
        probs = np.eye(num_classes)[pred_labels]

    # Convert multi-hot/one-hot target to index if necessary
    if targets.ndim == 2:
        target_labels = np.argmax(targets, axis=-1)
        onehot_targets = targets
    else:
        target_labels = targets
        onehot_targets = np.eye(num_classes)[target_labels]

    # Accuracy
    acc = float(np.mean(pred_labels == target_labels))

    # Precision, Recall, Macro-F1
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        target_labels,
        pred_labels,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    )

    # Micro-F1
    _, _, micro_f1, _ = precision_recall_fscore_support(
        target_labels,
        pred_labels,
        average="micro",
        zero_division=0,
    )

    # mAP from softmax probabilities per class
    aps: list[float] = []
    for c in range(num_classes):
        if np.sum(onehot_targets[:, c]) > 0:
            ap = average_precision_score(onehot_targets[:, c], probs[:, c])
            aps.append(float(ap))
    map_score = float(np.mean(aps)) if aps else 0.0

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "mAP": map_score,
    }


def compute_snr_breakdown(
    predictions: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    snrs: Tensor | np.ndarray,
    snr_levels: Sequence[float] = SNR_LEVELS,
    num_classes: int = 36,
) -> dict[str, dict[str, Any]]:
    """
    Computes metric breakdowns grouped by each SNR value in snr_levels.
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(snrs, Tensor):
        snrs = snrs.detach().cpu().numpy()

    breakdown: dict[str, dict[str, Any]] = {}
    for snr in snr_levels:
        mask = np.isclose(snrs, snr, atol=0.5)
        count = int(np.sum(mask))
        if count == 0:
            breakdown[f"{int(snr)}dB"] = {
                "n": 0,
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "mAP": 0.0,
            }
            continue

        sub_preds = predictions[mask]
        sub_targets = targets[mask]
        sub_metrics = compute_metrics(sub_preds, sub_targets, num_classes=num_classes)
        breakdown[f"{int(snr)}dB"] = {"n": count, **sub_metrics}

    return breakdown
