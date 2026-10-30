from __future__ import annotations

import numpy as np
import pytest
import torch

from src.router import EnergySNRRouter, NeuralRouter, estimate_snr_db


def test_estimate_snr_db() -> None:
    mixture = torch.ones(2, 16000)
    noise = torch.ones(2, 16000) * 0.1
    snr = estimate_snr_db(mixture, noise)
    assert snr.shape == (2,)
    assert torch.all(snr > 15.0)


def test_energy_snr_router_threshold() -> None:
    router = EnergySNRRouter(threshold_db=18.0)
    snrs = np.array([-5.0, 0.0, 10.0, 17.9, 18.0, 20.0])
    is_high = router.route(snrs)
    expected = np.array([False, False, False, False, True, True])
    np.testing.assert_array_equal(is_high, expected)


def test_energy_snr_router_estimate_and_route() -> None:
    router = EnergySNRRouter(threshold_db=18.0)
    mixture = torch.ones(1, 16000)
    loud_noise = torch.ones(1, 16000) * 0.8
    snrs, is_high = router.estimate_and_route(mixture, loud_noise)
    assert is_high.item() is False

    faint_noise = torch.ones(1, 16000) * 0.05
    snrs, is_high = router.estimate_and_route(mixture, faint_noise)
    assert is_high.item() is True


def test_neural_router_forward() -> None:
    router = NeuralRouter(in_features=4, hidden_dim=16)
    router.eval()
    mixture = torch.randn(3, 16000)
    noise = torch.randn(3, 16000) * 0.2
    alpha = router.route_soft(mixture, noise)
    assert alpha.shape == (3,)
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)
