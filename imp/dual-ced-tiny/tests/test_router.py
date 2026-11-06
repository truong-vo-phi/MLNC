"""
Unit tests for EnergySNRRouter and routing dynamics.
"""

import pytest
import torch

from src.models.dual_ced_tiny import DualCEDTiny
from src.router import EnergySNRRouter, estimate_snr_db


def test_estimate_snr_db():
    clean = torch.randn(2, 16000) * 1.0
    noise = torch.randn(2, 16000) * 0.1
    mixture = clean + noise
    est_snr = estimate_snr_db(mixture, noise)
    assert est_snr.shape == (2,)
    # Power ratio ~ (1.0 / 0.1)^2 = 100 -> ~20 dB
    assert (est_snr > 15.0).all()


def test_router_hard_threshold():
    router = EnergySNRRouter(threshold_db=18.0, mode="hard")
    wave = torch.randn(2, 64000)
    snrs = torch.tensor([10.0, 20.0])

    decision = router(wave, snrs_db=snrs)
    assert decision.alpha.shape == (2, 1)
    assert decision.alpha[0].item() == 0.0  # 10 dB < 18 dB -> Low expert
    assert decision.alpha[1].item() == 1.0  # 20 dB >= 18 dB -> High expert
    assert not decision.is_high_snr[0].item()
    assert decision.is_high_snr[1].item()


def test_dual_ced_routing_details():
    model = DualCEDTiny(num_classes=36)
    model.eval()

    dummy_wave = torch.randn(2, 64000) * 0.05
    dummy_snr = torch.tensor([5.0, 20.0])

    with torch.no_grad():
        blended, details = model(dummy_wave, snr_db=dummy_snr, return_details=True)

    assert blended.shape == (2, 36)
    assert "logits_low" in details
    assert "logits_high" in details
    assert "alpha" in details
    assert details["logits_low"].shape == (2, 36)
    assert details["logits_high"].shape == (2, 36)
