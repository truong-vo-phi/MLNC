from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .evaluate import Predictions, metrics_by_snr

PROB_EPS = 1e-8


@dataclass(frozen=True)
class EnsembleReport:
    predictions: Predictions
    metrics: dict[str, float]
    diversity: dict[str, float]


def average_predictions(members: Sequence[Predictions]) -> Predictions:
    if not members:
        raise ValueError("at least one ensemble member is required")
    first = members[0]
    for member in members[1:]:
        if member.sample_ids != first.sample_ids:
            raise ValueError("ensemble members were evaluated on different sample orders")
        if not np.array_equal(member.labels, first.labels):
            raise ValueError("ensemble members have different labels")
        if not np.allclose(member.snr_db, first.snr_db):
            raise ValueError("ensemble members have different SNR values")
    probabilities = np.mean([member.probabilities for member in members], axis=0)
    return Predictions(
        sample_ids=first.sample_ids,
        labels=first.labels,
        probabilities=probabilities.astype(np.float32),
        snr_db=first.snr_db,
    )


def ensemble_diversity(members: Sequence[Predictions]) -> dict[str, float]:
    if not members:
        raise ValueError("at least one ensemble member is required")
    labels = members[0].labels
    predictions = [member.probabilities.argmax(axis=-1) for member in members]
    individual = [float(np.mean(member == labels)) for member in predictions]
    oracle = float(np.mean(np.any(np.stack(predictions) == labels, axis=0)))
    disagreement: list[float] = []
    divergences: list[float] = []
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            disagreement.append(float(np.mean(predictions[left] != predictions[right])))
            p_left = np.clip(members[left].probabilities, PROB_EPS, 1.0)
            p_right = np.clip(members[right].probabilities, PROB_EPS, 1.0)
            divergences.append(float(np.mean(np.sum(p_left * (np.log(p_left) - np.log(p_right)), axis=-1))))
    ensemble = average_predictions(members)
    ensemble_top1 = float(np.mean(ensemble.probabilities.argmax(axis=-1) == labels))
    probabilities = np.clip(ensemble.probabilities, PROB_EPS, 1.0)
    entropy = float(np.mean(-np.sum(probabilities * np.log(probabilities), axis=-1)))
    return {
        "n_members": float(len(members)),
        "mean_individual_top1": float(np.mean(individual)),
        "oracle_top1": oracle,
        "ensemble_top1": ensemble_top1,
        "ensemble_gain": ensemble_top1 - float(np.mean(individual)),
        "disagreement": float(np.mean(disagreement)) if disagreement else 0.0,
        "mean_pairwise_kl": float(np.mean(divergences)) if divergences else 0.0,
        "mean_entropy": entropy,
    }


def evaluate_ensemble(members: Sequence[Predictions], snr_levels: Sequence[float]) -> EnsembleReport:
    predictions = average_predictions(members)
    return EnsembleReport(
        predictions=predictions,
        metrics=metrics_by_snr(predictions, snr_levels),
        diversity=ensemble_diversity(members),
    )
