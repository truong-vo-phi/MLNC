from __future__ import annotations

import numpy as np

from src.metrics import NINE_METRICS, ClassificationMetrics, snr_scores


def test_classification_metrics_perfect() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    probabilities = np.eye(4)[labels]
    result = ClassificationMetrics(num_classes=4)
    result.update(labels, probabilities)
    scores = result.compute()
    assert scores.top1_accuracy == 1.0
    assert scores.balanced_accuracy == 1.0
    assert scores.mAP == 1.0


def test_snr_scores_has_all_buckets() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    probabilities = np.eye(4)[labels]
    snrs = np.array([0, 0, 0, 0, 5, 5, 5, 5, 10, 10, 10, 10])
    scores = snr_scores(probabilities, labels, snrs, 4)
    assert scores["macro_acc"] == 1.0
    assert scores["mAP_macro"] == 1.0
    for name in NINE_METRICS:
        assert f"{name}_snr_0" in scores
        assert f"{name}_snr_5" in scores
        assert f"{name}_snr_10" in scores
        assert f"{name}_snr_mean" in scores
