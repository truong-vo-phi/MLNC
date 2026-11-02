"""
Evaluation Metrics for Multi-Class Audio Classification across SNR levels.
Computes: acc, precision, recall, macro_f1, micro_f1, mAP, and per-SNR breakdowns.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_topk_acc(preds: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """Tính toán độ chính xác Top-k (Top-k Accuracy)."""
    preds_arr = _to_numpy(preds)
    targets_arr = _to_numpy(targets)
    if targets_arr.ndim == 2:
        targets_arr = np.argmax(targets_arr, axis=-1)
    if len(targets_arr) == 0:
        return 0.0
    k = min(k, preds_arr.shape[-1])
    topk = np.argsort(preds_arr, axis=-1)[:, -k:]
    hits = np.any(topk == targets_arr[:, None], axis=-1)
    return float(np.mean(hits))


def compute_auc(preds: np.ndarray, targets: np.ndarray, num_classes: int = 36) -> float:
    """Tính toán Macro AUC-ROC trên các lớp có cả nhãn dương và âm."""
    preds_arr = _to_numpy(preds)
    targets_arr = _to_numpy(targets)
    if targets_arr.ndim == 1:
        y_true = np.zeros((len(targets_arr), num_classes), dtype=np.float32)
        for i, t in enumerate(targets_arr):
            if 0 <= int(t) < num_classes:
                y_true[i, int(t)] = 1.0
    else:
        y_true = targets_arr

    aucs = []
    for c in range(min(num_classes, preds_arr.shape[1])):
        pos_cnt = int(np.sum(y_true[:, c]))
        if 0 < pos_cnt < len(y_true):
            try:
                score = roc_auc_score(y_true[:, c], preds_arr[:, c])
                aucs.append(score)
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def compute_binary_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Computes Average Precision for a single class."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp_cum = np.cumsum(sorted_labels)
    total = np.arange(1, len(labels) + 1)
    precision = tp_cum / total
    boundaries = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    recall = tp_cum[boundaries] / n_pos
    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recall, precision[boundaries], strict=True):
        ap += (r - prev_recall) * p
        prev_recall = r
    return float(ap)


def compute_metrics(
    logits_or_probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    num_classes: int = 36,
) -> dict[str, float]:
    """
    Computes all core evaluation metrics:
    - acc (Top-1 Accuracy)
    - precision (Macro Precision)
    - recall (Macro Recall)
    - macro_f1 (Macro F1)
    - micro_f1 (Micro F1)
    - mAP (Mean Average Precision)
    """
    preds_raw = _to_numpy(logits_or_probs)
    y_true_raw = _to_numpy(targets)

    # Convert logits to probabilities if necessary
    if preds_raw.ndim == 2 and np.any(preds_raw < 0) or np.any(preds_raw > 1.0):
        # Apply softmax along class dim
        exps = np.exp(preds_raw - np.max(preds_raw, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
    else:
        probs = preds_raw

    # Convert one-hot to class index if needed
    if y_true_raw.ndim == 2:
        y_true_idx = np.argmax(y_true_raw, axis=-1)
        y_true_onehot = y_true_raw
    else:
        y_true_idx = y_true_raw.astype(np.int64)
        y_true_onehot = np.zeros((len(y_true_idx), num_classes), dtype=np.float32)
        for i, idx in enumerate(y_true_idx):
            if 0 <= idx < num_classes:
                y_true_onehot[i, idx] = 1.0

    y_pred_idx = np.argmax(probs, axis=-1)
    n_samples = len(y_true_idx)

    # 1. Accuracy
    acc = float(np.mean(y_pred_idx == y_true_idx)) if n_samples > 0 else 0.0

    # 2. Confusion matrix for precision, recall, f1
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp[c] = np.sum((y_pred_idx == c) & (y_true_idx == c))
        fp[c] = np.sum((y_pred_idx == c) & (y_true_idx != c))
        fn[c] = np.sum((y_pred_idx != c) & (y_true_idx == c))

    # Per-class precision & recall
    precisions = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recalls = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(tp),
        where=(precisions + recalls) > 0,
    )

    # Macro Precision, Recall, F1 (averaged over classes present in ground truth)
    active_classes = np.where((tp + fn) > 0)[0]
    if len(active_classes) > 0:
        macro_precision = float(np.mean(precisions[active_classes]))
        macro_recall = float(np.mean(recalls[active_classes]))
        macro_f1 = float(np.mean(f1_scores[active_classes]))
    else:
        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1_scores))

    # Micro F1 (equal to Accuracy in single-label multi-class setting)
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # 3. Mean Average Precision (mAP)
    aps = []
    for c in range(num_classes):
        target_c = y_true_onehot[:, c]
        if np.sum(target_c) > 0:
            ap_c = compute_binary_ap(target_c, probs[:, c])
            if not np.isnan(ap_c):
                aps.append(ap_c)
    mAP = float(np.mean(aps)) if aps else 0.0

    return {
        "acc": acc,
        "precision": macro_precision,
        "recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mAP": mAP,
    }


def compute_9_metrics_dict(
    logits_or_probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    alphas: np.ndarray | torch.Tensor | None = None,
    num_classes: int = 36,
) -> dict[str, float]:
    """
    Tính đầy đủ 9 chỉ số đánh giá cho Task 4:
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
    preds_raw = _to_numpy(logits_or_probs)
    y_true_raw = _to_numpy(targets)

    # Convert to probabilities
    if preds_raw.ndim == 2 and (np.any(preds_raw < 0) or np.any(preds_raw > 1.0)):
        exps = np.exp(preds_raw - np.max(preds_raw, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
    else:
        probs = preds_raw

    # Convert targets to class indices
    if y_true_raw.ndim == 2:
        y_true_idx = np.argmax(y_true_raw, axis=-1)
    else:
        y_true_idx = y_true_raw.astype(np.int64)

    y_pred_idx = np.argmax(probs, axis=-1)

    top1 = compute_topk_acc(probs, y_true_idx, k=1)
    top3 = compute_topk_acc(probs, y_true_idx, k=3)

    try:
        balanced = float(balanced_accuracy_score(y_true_idx, y_pred_idx))
    except Exception:
        balanced = top1

    base_metrics = compute_metrics(probs, y_true_raw, num_classes=num_classes)
    auc_score = compute_auc(probs, y_true_raw, num_classes=num_classes)

    mean_a = 0.50
    if alphas is not None:
        alphas_arr = _to_numpy(alphas).ravel()
        if len(alphas_arr) > 0:
            mean_a = float(np.mean(alphas_arr))

    return {
        "top1": top1,
        "top3": top3,
        "balanced_acc": balanced,
        "map": base_metrics["mAP"],
        "auc": auc_score,
        "precision": base_metrics["precision"],
        "recall": base_metrics["recall"],
        "f1": base_metrics["macro_f1"],
        "mean_alpha": mean_a,
        # Aliases for backward compatibility
        "acc": top1,
        "macro_f1": base_metrics["macro_f1"],
        "micro_f1": base_metrics["micro_f1"],
        "mAP": base_metrics["mAP"],
    }


def compute_snr_breakdown(
    probs: np.ndarray,
    targets: np.ndarray,
    snr_levels: np.ndarray,
    alphas: np.ndarray | None = None,
    num_classes: int = 36,
    target_snrs: Sequence[float] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
    use_9_metrics: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Computes all metrics broken down for each specific SNR level.
    """
    probs_arr = _to_numpy(probs)
    targets_arr = _to_numpy(targets)
    snrs_arr = _to_numpy(snr_levels).ravel()
    alphas_arr = _to_numpy(alphas).ravel() if alphas is not None else None

    breakdown: dict[str, dict[str, float]] = {}

    for snr in target_snrs:
        # Match within +/- 0.5 dB
        mask = np.abs(snrs_arr - snr) < 0.5
        count = int(np.sum(mask))
        if count > 0:
            sub_alphas = alphas_arr[mask] if alphas_arr is not None else None
            if use_9_metrics:
                metrics_snr = compute_9_metrics_dict(
                    probs_arr[mask], targets_arr[mask], alphas=sub_alphas, num_classes=num_classes
                )
            else:
                metrics_snr = compute_metrics(probs_arr[mask], targets_arr[mask], num_classes=num_classes)
            metrics_snr["count"] = count
            breakdown[f"{int(snr)}dB"] = metrics_snr
        else:
            default_metrics = {
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "mAP": 0.0,
                "count": 0,
            }
            if use_9_metrics:
                default_metrics.update({
                    "top1": 0.0,
                    "top3": 0.0,
                    "balanced_acc": 0.0,
                    "map": 0.0,
                    "auc": 0.0,
                    "f1": 0.0,
                    "mean_alpha": 0.50,
                })
            breakdown[f"{int(snr)}dB"] = default_metrics

    # Overall summary if use_9_metrics
    if use_9_metrics:
        all_metrics = compute_9_metrics_dict(
            probs_arr, targets_arr, alphas=alphas_arr, num_classes=num_classes
        )
        all_metrics["count"] = len(targets_arr)
        breakdown["all"] = all_metrics
        breakdown["overall"] = all_metrics

    return breakdown

