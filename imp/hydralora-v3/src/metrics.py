"""
Các chỉ số đánh giá (Evaluation Metrics) cho bài toán phân loại âm thanh 36 lớp phân rã theo từng dải SNR.

Bao gồm 9 metrics hoàn chỉnh:
1. Top-1 Accuracy (%)
2. Top-3 Accuracy (%)
3. Balanced Accuracy (%)
4. Macro mAP (%)
5. Macro AUC-ROC (%)
6. Macro Precision (%)
7. Macro Recall (%)
8. Macro F1-Score (%)
9. Mean Router Weight ᾱ (hoặc ECE)

Và phân tích chi tiết cho 6 mốc SNR: [-5dB, 0dB, 5dB, 10dB, 15dB, 20dB] + ALL (Tổng thể).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def compute_9_metrics_dict(
    preds: np.ndarray,
    targets: np.ndarray,
    alphas: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Tính đầy đủ 9 chỉ số cho một tập con hoặc toàn bộ tập đánh giá:
    1. top1: Top-1 Accuracy (%)
    2. top3: Top-3 Accuracy (%)
    3. balanced_acc: Balanced Accuracy (%)
    4. map: Macro mAP (%)
    5. auc: Macro AUC-ROC (%)
    6. precision: Macro Precision (%)
    7. recall: Macro Recall (%)
    8. f1: Macro F1-Score (%)
    9. mean_alpha: Mean Router Weight ᾱ
    """
    pred_labels = np.argmax(preds, axis=-1)

    top1 = compute_topk_acc(preds, targets, k=1)
    top3 = compute_topk_acc(preds, targets, k=3)
    balanced = float(balanced_accuracy_score(targets, pred_labels))
    m_ap = compute_map(preds, targets)
    auc = compute_auc(preds, targets)
    prec = float(precision_score(targets, pred_labels, average="macro", zero_division=0))
    rec = float(recall_score(targets, pred_labels, average="macro", zero_division=0))
    f1 = float(f1_score(targets, pred_labels, average="macro", zero_division=0))
    mean_a = float(np.mean(alphas)) if alphas is not None and len(alphas) > 0 else 0.50

    return {
        "top1": top1,
        "top3": top3,
        "balanced_acc": balanced,
        "map": m_ap,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_alpha": mean_a,
        "count": float(len(targets)),
    }


def compute_snr_breakdown(
    preds: np.ndarray,
    targets: np.ndarray,
    snrs: np.ndarray,
    alphas: np.ndarray | None = None,
    snr_bins: tuple[int, ...] = (-5, 0, 5, 10, 15, 20),
) -> dict[str, dict[str, float]]:
    """
    Tính toán phân rã hiệu năng chi tiết 9 metrics theo từng mức SNR cụ thể và tổng thể (ALL).
    """
    results: dict[str, dict[str, float]] = {}
    for snr in snr_bins:
        mask = np.isclose(snrs, snr, atol=1.0)
        if np.sum(mask) == 0:
            continue
        sub_preds = preds[mask]
        sub_targets = targets[mask]
        sub_alphas = alphas[mask] if alphas is not None else None
        results[f"{snr}dB"] = compute_9_metrics_dict(sub_preds, sub_targets, sub_alphas)

    # Tổng thể (ALL)
    results["all"] = compute_9_metrics_dict(preds, targets, alphas)
    results["overall"] = results["all"]  # alias
    return results
