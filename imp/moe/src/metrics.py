# -*- coding: utf-8 -*-
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_TOP_K = 3


@dataclass(frozen=True)
class ConfusionPair:
    """Cặp nhãn bị dự đoán nhầm (Ground Truth vs Prediction)"""
    true_label: str
    predicted_label: str
    count: int
    rate: float


@dataclass(frozen=True)
class MetricsResult:
    """Kết quả bộ 9 chỉ số đánh giá phân loại đa lớp (9 Core Classification Metrics)"""
    top1_accuracy: float       # Độ chính xác Top-1: tỷ lệ đoán đúng nhãn cao nhất
    top3_accuracy: float       # Độ chính xác Top-3: nhãn đúng nằm trong top 3 dự đoán
    balanced_accuracy: float   # Trung bình Recall trên từng lớp (tránh lệch do mất cân bằng dữ liệu)
    precision_macro: float     # Macro Precision: trung bình độ chính xác trên 36 lớp
    recall_macro: float        # Macro Recall: trung bình độ nhạy trên 36 lớp
    macro_f1: float            # Macro F1-score: trung bình điều hòa của Precision và Recall
    micro_f1: float            # Micro F1-score: tính gộp toàn cục (tương đương Overall Accuracy)
    mAP: float                 # Macro Mean Average Precision: diện tích dưới đường cong PR đa lớp
    macro_auc: float           # Macro ROC-AUC: diện tích dưới đường cong ROC đa lớp (One-vs-Rest)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _to_numpy(values: Any) -> np.ndarray:
    """Chuyển đổi tensor PyTorch hoặc mảng bất kỳ thành mảng NumPy chuẩn"""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """
    Xác định thứ hạng (ranks) của mảng giá trị, xử lý trường hợp bằng điểm (ties) bằng trung bình thứ hạng.
    Thuật toán dùng sắp xếp mergesort ổn định độ phức tạp O(N log N).
    """
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
    """
    Tính diện tích dưới đường cong ROC (ROC-AUC) nhị phân thông qua thống kê Mann-Whitney U.
    Công thức: U = sum(ranks của nhãn positive) - n_pos * (n_pos + 1) / 2
              AUC = U / (n_pos * n_neg)
    Tốc độ tính toán vượt trội và không cần phụ thuộc thư viện ngoài scikit-learn.
    """
    n_pos = int(labels.sum())
    n_neg = labels.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    pos_rank_sum = ranks[labels == 1].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _binary_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Tính Average Precision (AP) nhị phân bằng tích phân số học trên đường cong Precision-Recall.
    AP = sum_k (Recall_k - Recall_{k-1}) * Precision_k
    """
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
    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recall, precision[boundaries], strict=True):
        ap += (r - prev_recall) * p
        prev_recall = r
    return float(ap)


def confusion_matrix(y_true: Any, y_pred: Any, num_classes: int) -> np.ndarray:
    """Xây dựng ma trận nhầm lẫn kích thước [Num_Classes, Num_Classes] dùng bincount tốc độ cao"""
    y_true = _to_numpy(y_true).astype(np.int64).ravel()
    y_pred = _to_numpy(y_pred).astype(np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if not (np.all((y_true >= 0) & (y_true < num_classes)) and np.all((y_pred >= 0) & (y_pred < num_classes))):
        raise ValueError("labels out of range")
    counts = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def top_confusions(matrix: np.ndarray, class_names: Sequence[str], top_n: int = 10) -> list[ConfusionPair]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(class_names):
        raise ValueError("matrix must be square with one row per class name")
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    row_totals = matrix.sum(axis=1)
    order = np.argsort(-off_diagonal, axis=None, kind="mergesort")
    pairs = []
    for flat in order[:top_n]:
        true_index, predicted_index = divmod(int(flat), matrix.shape[1])
        count = int(off_diagonal[true_index, predicted_index])
        if count == 0:
            break
        rate = count / int(row_totals[true_index])
        pairs.append(ConfusionPair(class_names[true_index], class_names[predicted_index], count, rate))
    return pairs


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


class ClassificationMetrics:
    """
    Trình quản lý và tính toán trọn bộ 9 chỉ số phân loại âm thanh đa lớp.
    Hỗ trợ tích lũy theo từng batch (update) và xuất kết quả tổng hợp (compute).
    """
    def __init__(self, num_classes: int | None = None, top_k: int = _TOP_K) -> None:
        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.num_classes = num_classes
        self.top_k = top_k
        self._y_true: list[np.ndarray] = []
        self._y_score: list[np.ndarray] = []

    def reset(self) -> None:
        """Xóa sạch bộ đệm dữ liệu đã tích lũy"""
        self._y_true.clear()
        self._y_score.clear()

    def update(self, y_true: Any, y_score: Any) -> ClassificationMetrics:
        """Bổ sung batch nhãn thực tế và xác suất dự đoán vào bộ nhớ"""
        y_true = _to_numpy(y_true).astype(np.int64).ravel()
        y_score = _to_numpy(y_score).astype(np.float64)
        if y_score.ndim == 1:
            y_score = np.stack([1.0 - y_score, y_score], axis=1)
        if y_score.ndim != 2:
            raise ValueError("y_score must be a 1D or 2D array")
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true and y_score must have the same number of samples")
        self._y_true.append(y_true)
        self._y_score.append(y_score)
        return self

    @property
    def _num_classes(self) -> int:
        if self.num_classes is not None:
            return self.num_classes
        return self._y_score[0].shape[1]

    def confusion_matrix(self) -> np.ndarray:
        if not self._y_true:
            raise ValueError("no samples recorded, call update() first")
        y_score = np.concatenate(self._y_score)
        return confusion_matrix(np.concatenate(self._y_true), np.argmax(y_score, axis=1), self._num_classes)

    def compute(self) -> MetricsResult:
        """
        Tính toán 9 chỉ số chất lượng chính:
            - Top-1 & Top-k Hit Rate
            - TP, FP, FN ma trận nhầm lẫn để suy ra Precision, Recall, F1
            - One-vs-Rest AUC & Average Precision (mAP) trên toàn bộ các lớp
        """
        if not self._y_true:
            raise ValueError("no samples recorded, call update() first")
        y_true = np.concatenate(self._y_true)
        y_score = np.concatenate(self._y_score)
        num_classes = self._num_classes
        if not np.all((y_true >= 0) & (y_true < num_classes)):
            raise ValueError("y_true contains out-of-range labels")

        y_pred = np.argmax(y_score, axis=1)
        k = min(self.top_k, num_classes)
        top_k_pred = np.argsort(-y_score, axis=1)[:, :k]
        top_k_hit = (top_k_pred == y_true[:, None]).any(axis=1)

        true_counts = np.bincount(y_true, minlength=num_classes)
        pred_counts = np.bincount(y_pred, minlength=num_classes)
        tp = np.bincount(y_pred[y_pred == y_true], minlength=num_classes)
        fp = pred_counts - tp
        fn = true_counts - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)

        present = true_counts > 0
        present_precision = precision[present]
        present_recall = recall[present]
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                present_precision + present_recall > 0,
                2 * present_precision * present_recall / (present_precision + present_recall),
                0.0,
            )

        micro_tp = int(tp.sum())
        micro_f1 = micro_tp / y_true.shape[0]

        auc_values = []
        ap_values = []
        for c in range(num_classes):
            labels = (y_true == c).astype(np.int64)
            auc_values.append(_binary_auc(labels, y_score[:, c]))
            ap_values.append(_binary_ap(labels, y_score[:, c]))

        return MetricsResult(
            top1_accuracy=float(np.mean(y_pred == y_true)),
            top3_accuracy=float(np.mean(top_k_hit)),
            balanced_accuracy=_safe_mean(present_recall.tolist()),
            precision_macro=_safe_mean(present_precision.tolist()),
            recall_macro=_safe_mean(present_recall.tolist()),
            macro_f1=_safe_mean(f1.tolist()),
            micro_f1=float(micro_f1),
            mAP=_safe_mean([v for v in ap_values if not np.isnan(v)]),
            macro_auc=_safe_mean([v for v in auc_values if not np.isnan(v)]),
        )


NINE_METRICS = (
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


def snr_scores(probabilities: np.ndarray, labels: np.ndarray, snrs: np.ndarray, num_classes: int) -> dict[str, float]:
    """
    Tính toán chỉ số đánh giá toàn cục kết hợp phân rã chi tiết theo từng mốc SNR.
    
    Đầu ra bao gồm:
        - Chỉ số toàn cục (Top-1, Top-3, Balanced Acc, Macro mAP, Macro AUC, ...)
        - Các chỉ số trên từng mốc SNR cụ thể: -5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB
        - Trung bình cộng giữa các mốc SNR (ví dụ: top1_accuracy_snr_mean)
    """
    overall = ClassificationMetrics(num_classes=num_classes)
    overall.update(labels, probabilities)
    scores = overall.compute().to_dict()
    scores["acc"] = scores["top1_accuracy"]
    balanced, mean_ap = [], []
    per_metric: dict[str, list[float]] = {name: [] for name in NINE_METRICS}
    for snr in np.unique(snrs):
        mask = snrs == snr
        per_snr = ClassificationMetrics(num_classes=num_classes)
        per_snr.update(labels[mask], probabilities[mask])
        result = per_snr.compute()
        values = result.to_dict()
        scores[f"acc_snr_{int(snr)}"] = result.top1_accuracy
        scores[f"balanced_snr_{int(snr)}"] = result.balanced_accuracy
        scores[f"mAP_snr_{int(snr)}"] = result.mAP
        for name in NINE_METRICS:
            scores[f"{name}_snr_{int(snr)}"] = values[name]
            per_metric[name].append(values[name])
        balanced.append(result.balanced_accuracy)
        mean_ap.append(result.mAP)
    for name, series in per_metric.items():
        if series:
            scores[f"{name}_snr_mean"] = float(np.mean(series))
    scores["macro_acc"] = float(np.mean(balanced)) if balanced else scores["balanced_accuracy"]
    scores["mAP_macro"] = float(np.mean(mean_ap)) if mean_ap else scores["mAP"]
    return scores


class SnrBucketMeter:
    def __init__(self, high_levels: Sequence[float] = (15.0, 20.0)) -> None:
        self.high_levels = tuple(float(level) for level in high_levels)
        self.reset()

    def reset(self) -> None:
        self._sums: dict[float, float] = {}
        self._counts: dict[float, int] = {}

    def update(self, values: Any, snr_db: Any) -> SnrBucketMeter:
        scores = _to_numpy(values).astype(np.float64).reshape(-1)
        levels = _to_numpy(snr_db).astype(np.float64).reshape(-1)
        if scores.shape != levels.shape:
            raise ValueError(f"values {scores.shape} and snr_db {levels.shape} must match")
        for level in np.unique(levels):
            selected = scores[levels == level]
            key = float(level)
            self._sums[key] = self._sums.get(key, 0.0) + float(selected.sum())
            self._counts[key] = self._counts.get(key, 0) + int(selected.size)
        return self

    def means(self) -> dict[float, float]:
        return {level: self._sums[level] / self._counts[level] for level in sorted(self._sums)}

    def compute(self, prefix: str) -> dict[str, float]:
        if not self._counts:
            raise ValueError("no samples recorded, call update() first")
        means = self.means()
        result = {prefix: sum(self._sums.values()) / sum(self._counts.values())}
        result.update({f"{prefix}_{level:g}": mean for level, mean in means.items()})
        high = [means[level] for level in self.high_levels if level in means]
        if high:
            result[f"{prefix}_hi"] = sum(high) / len(high)
        return result
