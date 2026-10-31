"""
Evaluation Metrics with Multi-Class Metrics and SNR Stratification.

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
    - BEATs: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023. arXiv:2212.09058.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

_TOP_K = 3


@dataclass(frozen=True)
class MetricsResult:
    top1_accuracy: float
    top3_accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    macro_f1: float
    micro_f1: float
    mAP: float
    macro_auc: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = labels.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    pos_rank_sum = ranks[labels == 1].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _binary_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp_cum = np.cumsum(sorted_labels)
    total = np.arange(1, labels.shape[0] + 1)
    precision = tp_cum / total
    boundaries = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    recall = tp_cum[boundaries] / n_pos
    precision_at_boundaries = precision[boundaries]
    recall_diff = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision_at_boundaries * recall_diff))


class ClassificationMetrics:
    def __init__(self, num_classes: int = 36) -> None:
        self.num_classes = num_classes
        self._predictions: list[np.ndarray] = []
        self._probabilities: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, targets: Any, probabilities: Any) -> None:
        targets_np = _to_numpy(targets).astype(np.int64)
        probs_np = _to_numpy(probabilities).astype(np.float64)
        if targets_np.ndim != 1:
            raise ValueError(f"targets must be 1D, got {targets_np.ndim}")
        if probs_np.ndim != 2 or probs_np.shape[1] != self.num_classes:
            raise ValueError(f"probabilities must be [N, {self.num_classes}], got {probs_np.shape}")
        preds_np = np.argmax(probs_np, axis=1)
        self._targets.append(targets_np)
        self._probabilities.append(probs_np)
        self._predictions.append(preds_np)

    def compute(self) -> MetricsResult:
        if not self._targets:
            raise ValueError("No samples have been recorded.")
        targets = np.concatenate(self._targets, axis=0)
        probs = np.concatenate(self._probabilities, axis=0)
        preds = np.concatenate(self._predictions, axis=0)
        num_samples = targets.shape[0]

        # Top-1 Accuracy
        correct_top1 = targets == preds
        top1_acc = float(np.mean(correct_top1))

        # Top-3 Accuracy
        top3_indices = np.argsort(-probs, axis=1)[:, :_TOP_K]
        top3_acc = float(np.mean(np.any(top3_indices == targets[:, None], axis=1)))

        # Confusion Matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(targets, preds):
            cm[t, p] += 1

        recalls, precisions = [], []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

        balanced_acc = float(np.mean(recalls))
        macro_prec = float(np.mean(precisions))
        macro_rec = float(np.mean(recalls))
        macro_f1 = (
            float(2 * macro_prec * macro_rec / (macro_prec + macro_rec))
            if (macro_prec + macro_rec) > 0
            else 0.0
        )
        micro_f1 = top1_acc

        # One-vs-Rest for AUC and mAP
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(num_samples), targets] = 1.0

        ap_list, auc_list = [], []
        for c in range(self.num_classes):
            ap = _binary_ap(one_hot[:, c], probs[:, c])
            if not np.isnan(ap):
                ap_list.append(ap)
            auc = _binary_auc(one_hot[:, c], probs[:, c])
            if not np.isnan(auc):
                auc_list.append(auc)

        mean_ap = float(np.mean(ap_list)) if ap_list else 0.0
        macro_auc = float(np.mean(auc_list)) if auc_list else 0.0

        return MetricsResult(
            top1_accuracy=top1_acc,
            top3_accuracy=top3_acc,
            balanced_accuracy=balanced_acc,
            precision_macro=macro_prec,
            recall_macro=macro_rec,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            mAP=mean_ap,
            macro_auc=macro_auc,
        )


def compute_snr_breakdown(
    probabilities: np.ndarray,
    targets: np.ndarray,
    snrs: np.ndarray,
    num_classes: int = 36,
) -> dict[str, float]:
    """
    Compute overall classification metrics and breakdown across SNR levels (-5 to 20 dB).
    """
    overall = ClassificationMetrics(num_classes=num_classes)
    overall.update(targets, probabilities)
    scores = overall.compute().to_dict()
    scores["acc"] = scores["top1_accuracy"]

    balanced_list, map_list = [], []
    for snr in np.unique(snrs):
        mask = snrs == snr
        snr_int = int(round(float(snr)))
        per_snr = ClassificationMetrics(num_classes=num_classes)
        per_snr.update(targets[mask], probabilities[mask])
        res = per_snr.compute()
        scores[f"acc_snr_{snr_int}"] = res.top1_accuracy
        scores[f"top3_snr_{snr_int}"] = res.top3_accuracy
        scores[f"balanced_snr_{snr_int}"] = res.balanced_accuracy
        scores[f"mAP_snr_{snr_int}"] = res.mAP
        scores[f"auc_snr_{snr_int}"] = res.macro_auc
        balanced_list.append(res.balanced_accuracy)
        map_list.append(res.mAP)

    scores["macro_acc"] = float(np.mean(balanced_list)) if balanced_list else scores["balanced_accuracy"]
    scores["mAP_macro"] = float(np.mean(map_list)) if map_list else scores["mAP"]
    return scores
