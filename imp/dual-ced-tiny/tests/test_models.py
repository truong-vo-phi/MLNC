"""
Unit tests for Dual-CED-Tiny and single expert models.
"""

import pytest
import torch

from src.models import build_model, list_available_models
from src.utils import count_parameters


@pytest.mark.parametrize("model_name", list_available_models())
def test_model_instantiation_and_waveform_forward(model_name: str, device: torch.device):
    model = build_model(model_name, num_classes=36).to(device)
    model.eval()

    dummy_wave = torch.randn(2, 64000, device=device) * 0.05
    dummy_snr = torch.tensor([0.0, 20.0], device=device)
    with torch.no_grad():
        logits = model(dummy_wave, snr_db=dummy_snr)

    assert logits.shape == (2, 36)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize("model_name", list_available_models())
def test_model_spectrogram_forward(model_name: str, device: torch.device):
    model = build_model(model_name, num_classes=36).to(device)
    model.eval()

    dummy_spec = torch.randn(2, 1, 64, 398, device=device)
    dummy_snr = torch.tensor([-5.0, 15.0], device=device)
    with torch.no_grad():
        logits = model(dummy_spec, snr_db=dummy_snr)

    assert logits.shape == (2, 36)
    assert not torch.isnan(logits).any()


def test_parameter_counts():
    dual_model = build_model("dual_ced_tiny")
    total_dual, _ = count_parameters(dual_model)
    # Dual-CED-Tiny: ~10.79M - 10.84M params
    assert 10_000_000 <= total_dual <= 11_500_000, f"Dual-CED-Tiny params: {total_dual}"

    low_expert = build_model("ced_tiny_low")
    total_low, _ = count_parameters(low_expert)
    # Single CED-Tiny: ~5.40M params
    assert 5_000_000 <= total_low <= 6_000_000, f"CED-Tiny Low params: {total_low}"

    high_expert = build_model("ced_tiny_high")
    total_high, _ = count_parameters(high_expert)
    assert 5_000_000 <= total_high <= 6_000_000, f"CED-Tiny High params: {total_high}"
