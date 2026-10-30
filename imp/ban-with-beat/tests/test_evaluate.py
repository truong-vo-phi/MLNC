from __future__ import annotations

import numpy as np
from ban_beat.evaluate import (
    EVAL_METRICS,
    Predictions,
    load_predictions,
    metrics_by_snr,
    metrics_table,
    plot_snr_comparison,
    save_predictions,
    snr_curve,
)


def _predictions() -> Predictions:
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float32)[labels]
    probabilities[3] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return Predictions(
        sample_ids=("a", "b", "c", "d"),
        labels=labels,
        probabilities=probabilities,
        snr_db=np.array([-5.0, -5.0, 5.0, 15.0], dtype=np.float32),
    )


def test_metrics_by_snr_are_unweighted_over_levels() -> None:
    scores = metrics_by_snr(_predictions(), (-5.0, 0.0, 5.0, 15.0))
    assert scores["top1_accuracy"] == 0.75
    assert scores["top1_accuracy_snr_-5"] == 1.0
    assert scores["top1_accuracy_snr_5"] == 1.0
    assert scores["top1_accuracy_snr_15"] == 0.0
    assert scores["top1_accuracy_snr"] == 2.0 / 3.0
    assert scores["top1_accuracy_hi"] == 0.0
    assert scores["macro_f1_snr"] > 0.0


def test_missing_snr_level_is_skipped() -> None:
    scores = metrics_by_snr(_predictions(), (-5.0, 20.0))
    assert "top1_accuracy_snr_-5" in scores
    assert "top1_accuracy_snr_20" not in scores
    assert scores["top1_accuracy_snr"] == 1.0


def test_snr_curve_has_requested_levels() -> None:
    curve = snr_curve(_predictions(), (-5.0, 5.0, 15.0))
    assert sorted(curve["top1_accuracy"]) == [-5.0, 5.0, 15.0]


def test_save_and_load_predictions(tmp_path) -> None:
    predictions = _predictions()
    path = save_predictions(tmp_path / "predictions.npz", predictions)
    loaded = load_predictions(path)
    assert loaded.sample_ids == predictions.sample_ids
    np.testing.assert_array_equal(loaded.labels, predictions.labels)
    np.testing.assert_allclose(loaded.probabilities, predictions.probabilities)
    np.testing.assert_allclose(loaded.snr_db, predictions.snr_db)


def test_plot_snr_comparison(tmp_path) -> None:
    output = plot_snr_comparison({"member": _predictions()}, (-5.0, 5.0, 15.0), tmp_path / "snr.png")
    assert output.is_file()
    assert output.stat().st_size > 0


def test_metrics_table_has_each_snr_and_all() -> None:
    rng = np.random.default_rng(0)
    count = 60
    probabilities = rng.random((count, 36)).astype(np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = Predictions(
        sample_ids=tuple(f"s{i}" for i in range(count)),
        labels=np.arange(count) % 36,
        probabilities=probabilities,
        snr_db=np.tile(np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float32), count // 6),
    )
    rows = metrics_table(predictions, (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0))
    assert [row["snr"] for row in rows] == ["-5", "0", "5", "10", "15", "20", "all"]
    assert all(set(row) == {"snr", *EVAL_METRICS} for row in rows)
    assert len(EVAL_METRICS) == 9
