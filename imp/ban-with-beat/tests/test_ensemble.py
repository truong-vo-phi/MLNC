from __future__ import annotations

import numpy as np
import pytest
from ban_beat.ensemble import average_predictions, ensemble_diversity, evaluate_ensemble
from ban_beat.evaluate import Predictions


def _member(flip: int = 0) -> Predictions:
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float32)[labels]
    if flip:
        probabilities[2:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return Predictions(
        sample_ids=("a", "b", "c", "d"),
        labels=labels,
        probabilities=probabilities,
        snr_db=np.array([-5.0, 0.0, 5.0, 15.0], dtype=np.float32),
    )


def test_average_predictions_is_mean() -> None:
    left, right = _member(), _member(flip=1)
    averaged = average_predictions([left, right])
    np.testing.assert_allclose(averaged.probabilities, (left.probabilities + right.probabilities) / 2)
    assert averaged.sample_ids == left.sample_ids


def test_average_rejects_different_orders() -> None:
    left = _member()
    right = Predictions(
        sample_ids=("b", "a", "c", "d"),
        labels=left.labels,
        probabilities=left.probabilities,
        snr_db=left.snr_db,
    )
    with pytest.raises(ValueError, match="sample orders"):
        average_predictions([left, right])


def test_diversity_reports_gain_and_oracle() -> None:
    report = ensemble_diversity([_member(), _member(flip=1)])
    assert report["n_members"] == 2.0
    assert report["disagreement"] == 0.25
    assert report["oracle_top1"] == 1.0
    assert report["ensemble_top1"] == 0.75
    assert report["ensemble_gain"] == -0.125
    assert report["mean_pairwise_kl"] > 0.0


def test_single_member_ensemble_is_identity() -> None:
    member = _member()
    report = evaluate_ensemble([member], (-5.0, 0.0, 5.0, 15.0))
    np.testing.assert_allclose(report.predictions.probabilities, member.probabilities)
    assert report.diversity["disagreement"] == 0.0
    assert report.metrics["top1_accuracy"] == 1.0
