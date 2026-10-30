from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import ClassificationMetrics
from src.utils.plots import BLUE, ORANGE

from .runtime import autocast_context

HIGH_SNR = 15.0
DEFAULT_METRICS = ("acc", "top1_accuracy", "macro_f1", "mAP")


@dataclass(frozen=True)
class Predictions:
    """Chứa kết quả dự đoán của mô hình trên một tập dữ liệu."""

    sample_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    snr_db: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.labels.shape[0] != self.probabilities.shape[0]
            or self.labels.shape[0] != self.snr_db.shape[0]
        ):
            raise ValueError("labels, probabilities and snr_db must have the same number of rows")
        if self.labels.shape[0] != len(self.sample_ids):
            raise ValueError("sample_ids and prediction rows differ")
        if self.probabilities.ndim != 2:
            raise ValueError("probabilities must be a 2D array")


@torch.no_grad()
def collect_predictions(
    predict_fn: Callable[[Tensor], Tensor],
    loader: DataLoader,
    device: torch.device,
    amp: str = "none",
    desc: str = "evaluate",
) -> Predictions:
    """Thu thập toàn bộ probabilities và nhãn từ một DataLoader."""
    sample_ids: list[str] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    snrs: list[np.ndarray] = []

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        waveform = batch["waveform"].to(device)
        with autocast_context(device, amp):
            logits_or_probs = predict_fn(waveform)
        if hasattr(logits_or_probs, "logits"):
            logits = logits_or_probs.logits
            probs = torch.softmax(logits.float(), dim=-1)
        elif logits_or_probs.ndim == 2:
            # Nếu tensor chưa chuẩn hóa xác suất thì softmax
            if (
                torch.any(logits_or_probs < 0.0)
                or not torch.allclose(
                    logits_or_probs.sum(dim=-1),
                    torch.ones_like(logits_or_probs[:, 0]),
                    atol=1e-3,
                )
            ):
                probs = torch.softmax(logits_or_probs.float(), dim=-1)
            else:
                probs = logits_or_probs.float()
        else:
            probs = torch.softmax(logits_or_probs.float(), dim=-1)

        probabilities.append(probs.cpu().numpy())

        target = batch.get("target")
        if target is not None:
            if target.ndim == 2:
                labels.append(target.argmax(dim=1).cpu().numpy())
            else:
                labels.append(target.cpu().numpy())
        else:
            labels.append(np.zeros(len(probs), dtype=np.int64))

        if "snr_db" in batch:
            snrs.append(batch["snr_db"].cpu().numpy())
        else:
            snrs.append(np.zeros(len(probs), dtype=np.float32))

        if "sample_id" in batch:
            sample_ids.extend(str(s) for s in batch["sample_id"])
        else:
            sample_ids.extend(f"sample_{len(sample_ids) + i}" for i in range(len(probs)))

    if not probabilities:
        raise ValueError("loader produced no batches")

    return Predictions(
        sample_ids=tuple(sample_ids),
        labels=np.concatenate(labels).astype(np.int64),
        probabilities=np.concatenate(probabilities).astype(np.float32),
        snr_db=np.concatenate(snrs).astype(np.float32),
    )


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Tính các metrics chuẩn phân loại từ labels và xác suất dự đoán."""
    metrics = ClassificationMetrics(num_classes=probabilities.shape[1])
    metrics.update(labels, probabilities)
    res = metrics.compute().to_dict()
    res["acc"] = res["top1_accuracy"]
    return res


def metrics_by_snr(
    predictions: Predictions,
    snr_levels: Sequence[float],
    high_snr: float = HIGH_SNR,
) -> dict[str, float]:
    """Tính toán chi tiết metrics tổng quát và phân rã theo từng mức SNR."""
    scores = compute_metrics(predictions.labels, predictions.probabilities)
    per_f1: list[float] = []
    per_acc: list[float] = []
    per_map: list[float] = []

    for level in snr_levels:
        mask = np.isclose(predictions.snr_db, float(level))
        if not mask.any():
            continue
        sub_res = compute_metrics(predictions.labels[mask], predictions.probabilities[mask])
        suffix = f"{level:g}"
        scores[f"top1_accuracy_snr_{suffix}"] = sub_res["top1_accuracy"]
        scores[f"macro_f1_snr_{suffix}"] = sub_res["macro_f1"]
        scores[f"mAP_snr_{suffix}"] = sub_res["mAP"]
        per_acc.append(sub_res["top1_accuracy"])
        per_f1.append(sub_res["macro_f1"])
        per_map.append(sub_res["mAP"])

    if per_acc:
        scores["top1_accuracy_snr"] = float(np.mean(per_acc))
        scores["macro_f1_snr"] = float(np.mean(per_f1))
        scores["mAP_snr"] = float(np.mean(per_map))

    return scores


def compare_raw_and_ema(
    raw_scores: dict[str, float],
    ema_scores: dict[str, float],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, dict[str, float]]:
    """So sánh trực tiếp kết quả giữa Raw Model và EMA Model kèm delta cải thiện."""
    comparison: dict[str, dict[str, float]] = {}
    for metric in metrics:
        if metric in raw_scores and metric in ema_scores:
            raw_val = raw_scores[metric]
            ema_val = ema_scores[metric]
            delta = ema_val - raw_val
            comparison[metric] = {
                "raw": raw_val,
                "ema": ema_val,
                "delta": delta,
                "improvement_pct": (delta / max(raw_val, 1e-6)) * 100.0,
            }
    return comparison


def format_comparison_table(
    split_name: str,
    raw_scores: dict[str, float],
    ema_scores: dict[str, float],
) -> str:
    """Tạo bảng markdown so sánh đẹp mắt giữa Raw Model và EMA Model."""
    lines = [
        f"### Kết quả so sánh trên tập: `{split_name}`",
        "",
        "| Metric | Raw Model | EMA Shadow Model | Delta (EMA - Raw) | Cải thiện (%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    comparison = compare_raw_and_ema(raw_scores, ema_scores)
    for metric, data in comparison.items():
        delta_sign = "+" if data["delta"] >= 0 else ""
        lines.append(
            f"| **{metric}** | {data['raw']:.4f} | {data['ema']:.4f} | {delta_sign}{data['delta']:.4f} | {delta_sign}{data['improvement_pct']:.2f}% |"
        )
    return "\n".join(lines)
