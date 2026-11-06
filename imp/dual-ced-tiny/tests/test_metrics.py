"""
Unit tests for 6 core metrics and SNR breakdown in Dual-CED-Tiny.
"""

import numpy as np
import torch

from src.metrics import compute_metrics, compute_snr_breakdown


def test_compute_metrics_perfect_prediction():
    # 3 classes, 3 samples perfect match
    targets = np.array([0, 1, 2])
    logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])

    metrics = compute_metrics(logits, targets, num_classes=3)
    assert abs(metrics["acc"] - 1.0) < 1e-6
    assert abs(metrics["precision"] - 1.0) < 1e-6
    assert abs(metrics["recall"] - 1.0) < 1e-6
    assert abs(metrics["macro_f1"] - 1.0) < 1e-6
    assert abs(metrics["micro_f1"] - 1.0) < 1e-6
    assert abs(metrics["mAP"] - 1.0) < 1e-6


def test_snr_breakdown_computation():
    targets = np.array([0, 1, 2, 0])
    logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],  # Error on class 2
        [10.0, 0.0, 0.0],
    ])
    snrs = np.array([-5.0, 0.0, 0.0, 20.0])

    breakdown = compute_snr_breakdown(logits, targets, snrs, snr_levels=[-5.0, 0.0, 20.0], num_classes=3)
    assert "-5dB" in breakdown
    assert "0dB" in breakdown
    assert "20dB" in breakdown
    assert breakdown["-5dB"]["n"] == 1
    assert breakdown["-5dB"]["acc"] == 1.0
    assert breakdown["0dB"]["n"] == 2
    assert breakdown["0dB"]["acc"] == 0.5  # 1 of 2 correct
    assert breakdown["20dB"]["n"] == 1
    assert breakdown["20dB"]["acc"] == 1.0
