"""
Các chỉ số đánh giá (Evaluation Metrics) cho bài toán phân loại âm thanh 36 lớp phân rã theo từng dải SNR.

Bao gồm:
- Top-1 và Top-3 Accuracy
- Macro mAP (Mean Average Precision)
- Macro AUC (Area Under ROC Curve)
- Phân tích chi tiết theo dải SNR (-5dB, 0dB, 5dB, 10dB, 15dB, 20dB)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_topk_acc(preds: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """Tính toán độ chính xác Top-k (Top-k Accuracy)."""
    topk = np.argsort(preds, axis=-1)[:, -k:]
    hits = np.any(topk == targets[:, None], axis=-1)
    return float(np.mean(hits))


def compute_map(preds: np.ndarray, targets: np.ndarray) -> float:
    """Tính toán Macro mAP (Mean Average Precision) trên toàn bộ 36 lớp."""
    num_classes = preds.shape[1]
    if targets.ndim == 1:
        y_true = np.zeros_like(preds)
        for i, t in enumerate(targets):
            y_true[i, t] = 1.0
    else:
        y_true = targets

    aps = []
    for c in range(num_classes):
        if np.sum(y_true[:, c]) > 0:
            aps.append(average_precision_score(y_true[:, c], preds[:, c]))
    return float(np.mean(aps)) if aps else 0.0


def compute_auc(preds: np.ndarray, targets: np.ndarray) -> float:
    """Tính toán Macro AUC (ROC Area Under Curve) trên toàn bộ các lớp có cả mẫu âm và mẫu dương."""
    num_classes = preds.shape[1]
    if targets.ndim == 1:
        y_true = np.zeros_like(preds)
        for i, t in enumerate(targets):
            y_true[i, t] = 1.0
    else:
        y_true = targets

    aucs = []
    for c in range(num_classes):
        pos_cnt = int(np.sum(y_true[:, c]))
        if 0 < pos_cnt < len(y_true):
            aucs.append(roc_auc_score(y_true[:, c], preds[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def compute_snr_breakdown(
    preds: np.ndarray,
    targets: np.ndarray,
    snrs: np.ndarray,
    snr_bins: tuple[int, ...] = (-5, 0, 5, 10, 15, 20),
) -> dict[str, dict[str, float]]:
    """
    Tính toán phân rã hiệu năng chi tiết (Top-1, Top-3, mAP) theo từng mức SNR cụ thể và tổng thể.
    """
    results: dict[str, dict[str, float]] = {}
    for snr in snr_bins:
        mask = np.isclose(snrs, snr, atol=1.0)
        if np.sum(mask) == 0:
            continue
        sub_preds = preds[mask]
        sub_targets = targets[mask]
        results[f"{snr}dB"] = {
            "top1": compute_topk_acc(sub_preds, sub_targets, k=1),
            "top3": compute_topk_acc(sub_preds, sub_targets, k=3),
            "map": compute_map(sub_preds, sub_targets),
            "count": float(np.sum(mask)),
        }

    results["overall"] = {
        "top1": compute_topk_acc(preds, targets, k=1),
        "top3": compute_topk_acc(preds, targets, k=3),
        "map": compute_map(preds, targets),
        "auc": compute_auc(preds, targets),
        "count": float(len(targets)),
    }
    return results
