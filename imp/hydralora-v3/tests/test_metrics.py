import sys
from pathlib import Path
HYDRALORA_DIR = Path(__file__).resolve().parent.parent
for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]
if str(HYDRALORA_DIR) not in sys.path:
    sys.path.insert(0, str(HYDRALORA_DIR))

import numpy as np
import pytest

from src.metrics import compute_9_metrics_dict, compute_snr_breakdown, compute_topk_acc


def test_compute_topk_acc():
    preds = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.3, 0.6],
    ])
    targets = np.array([0, 1, 2])
    assert compute_topk_acc(preds, targets, k=1) == 1.0

    targets_wrong = np.array([1, 0, 0])
    assert compute_topk_acc(preds, targets_wrong, k=1) == 0.0


def test_compute_snr_breakdown():
    np.random.seed(42)
    n = 120
    preds = np.random.dirichlet(np.ones(36), size=n)
    targets = np.random.randint(0, 36, size=n)
    snrs = np.random.choice([-5, 0, 5, 10, 15, 20], size=n)

    res = compute_snr_breakdown(preds, targets, snrs)
    assert "all" in res
    assert "top1" in res["all"]
    assert "map" in res["all"]
    assert "auc" in res["all"]
