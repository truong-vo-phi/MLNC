import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from src.metrics import ClassificationMetrics


@pytest.fixture(scope="module")
def multiclass():
    rng = np.random.default_rng(0)
    n, c = 5000, 7
    y_true = rng.integers(0, c, n)
    logits = rng.normal(0, 1, (n, c))
    y_score = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return rng, c, y_true, y_score


def test_matches_sklearn(multiclass):
    _, c, y_true, y_score = multiclass
    y_pred = y_score.argmax(axis=1)
    metrics = ClassificationMetrics(num_classes=c)
    metrics.update(y_true[:2500], y_score[:2500]).update(y_true[2500:], y_score[2500:])
    r = metrics.compute()

    assert np.isclose(r.top1_accuracy, accuracy_score(y_true, y_pred))
    assert np.isclose(r.top3_accuracy, top_k_accuracy_score(y_true, y_score, k=3, labels=range(c)))
    assert np.isclose(r.balanced_accuracy, balanced_accuracy_score(y_true, y_pred))
    assert np.isclose(r.precision_macro, precision_score(y_true, y_pred, average="macro", labels=np.unique(y_true)))
    assert np.isclose(r.recall_macro, recall_score(y_true, y_pred, average="macro"))
    assert np.isclose(r.macro_f1, f1_score(y_true, y_pred, average="macro"))
    assert np.isclose(r.micro_f1, f1_score(y_true, y_pred, average="micro"))
    assert np.isclose(r.mAP, np.mean([average_precision_score(y_true == k, y_score[:, k]) for k in range(c)]))
    assert np.isclose(r.macro_auc, roc_auc_score(y_true, y_score, average="macro", multi_class="ovr", labels=range(c)))


def test_tied_scores(multiclass):
    _, c, y_true, y_score = multiclass
    tie_score = np.round(y_score, 1)
    tie_score = tie_score / tie_score.sum(axis=1, keepdims=True)
    r = ClassificationMetrics(num_classes=c).update(y_true, tie_score).compute()

    assert np.isclose(
        r.macro_auc, roc_auc_score(y_true, tie_score, average="macro", multi_class="ovr", labels=range(c))
    )
    assert np.isclose(r.mAP, np.mean([average_precision_score(y_true == k, tie_score[:, k]) for k in range(c)]))


def test_binary_scores(multiclass):
    rng = multiclass[0]
    b_true = rng.integers(0, 2, 300)
    b_score = rng.random(300)
    r = ClassificationMetrics(num_classes=2).update(b_true, b_score).compute()

    assert np.isclose(r.macro_auc, roc_auc_score(b_true, b_score))
    expected_map = np.mean([average_precision_score(b_true == k, [1 - b_score, b_score][k]) for k in range(2)])
    assert np.isclose(r.mAP, expected_map)


def test_compute_without_update_raises():
    with pytest.raises(ValueError):
        ClassificationMetrics().compute()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ClassificationMetrics(num_classes=1)
    with pytest.raises(ValueError):
        ClassificationMetrics(top_k=0)
