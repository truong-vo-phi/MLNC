"""
Unit tests for evaluation metrics and SNR breakdown.
"""

import numpy as np
import pytest
import torch

from src.metrics import compute_metrics, compute_snr_breakdown


def test_compute_metrics_perfect_predictions():
    num_classes = 5
    # Perfect predictions: diagonal logits
    targets = np.array([0, 1, 2, 3, 4])
    logits = np.eye(num_classes) * 10.0

    metrics = compute_metrics(logits, targets, num_classes=num_classes)

    assert pytest.approx(metrics["acc"], 1e-4) == 1.0
    assert pytest.approx(metrics["precision"], 1e-4) == 1.0
    assert pytest.approx(metrics["recall"], 1e-4) == 1.0
    assert pytest.approx(metrics["macro_f1"], 1e-4) == 1.0
    assert pytest.approx(metrics["micro_f1"], 1e-4) == 1.0
    assert pytest.approx(metrics["mAP"], 1e-4) == 1.0


def test_compute_metrics_imperfect():
    num_classes = 4
    targets = np.array([0, 1, 2, 3])
    # Class 0 predicted as 1
    logits = np.array([
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.9],
    ])
    metrics = compute_metrics(logits, targets, num_classes=num_classes)

    assert 0.0 < metrics["acc"] < 1.0
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert "precision" in metrics
    assert "recall" in metrics
    assert "mAP" in metrics


def test_compute_snr_breakdown():
    num_classes = 4
    targets = np.array([0, 1, 2, 3, 0, 1])
    probs = np.eye(num_classes)[[0, 1, 2, 3, 0, 1]]
    snrs = np.array([-5.0, -5.0, 0.0, 0.0, 20.0, 20.0])

    breakdown = compute_snr_breakdown(probs, targets, snrs, num_classes=num_classes, target_snrs=(-5.0, 0.0, 20.0))

    assert "-5dB" in breakdown
    assert "0dB" in breakdown
    assert "20dB" in breakdown
    assert breakdown["-5dB"]["acc"] == 1.0

