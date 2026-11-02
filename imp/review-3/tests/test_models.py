"""
Unit tests for all 9 models in Review-3.
"""

import pytest
import torch

from src.models import build_model, list_available_models


@pytest.mark.parametrize("model_name", list_available_models())
def test_model_instantiation_and_forward(model_name: str):
    """Verifies each model instantiates properly and handles waveform input."""
    device = torch.device("cpu")
    model = build_model(model_name, num_classes=36).to(device)
    model.eval()

    # Test raw waveform input: Batch=2, duration=4.0s @ 16kHz = 64,000 samples
    dummy_wave = torch.randn(2, 64000, device=device)
    with torch.no_grad():
        logits = model(dummy_wave)

    assert logits.shape == (2, 36), f"{model_name} failed raw waveform output shape check"
    assert not torch.isnan(logits).any(), f"{model_name} produced NaN in logits"


@pytest.mark.parametrize("model_name", ["pann_cnn10", "ced_tiny", "efficient_mn10"])
def test_model_spectrogram_input(model_name: str):
    """Verifies model directly accepts precomputed 4D spectrogram (B, 1, F, T)."""
    device = torch.device("cpu")
    n_mels = 64 if "ced" in model_name else 128
    model = build_model(model_name, num_classes=36, n_mels=n_mels).to(device)
    model.eval()

    dummy_spec = torch.randn(2, 1, n_mels, 398, device=device)
    with torch.no_grad():
        logits = model(dummy_spec)

    assert logits.shape == (2, 36)
    assert not torch.isnan(logits).any()


def test_parameter_counts():
    """Sanity check parameter counts for key models."""
    ced_tiny = build_model("ced_tiny")
    total_p, _ = ced_tiny.count_parameters()
    assert 5_000_000 <= total_p <= 6_000_000, f"CED-Tiny params out of expected range: {total_p}"

    cnn10 = build_model("pann_cnn10")
    total_p, _ = cnn10.count_parameters()
    assert 4_000_000 <= total_p <= 6_000_000, f"CNN10 params out of expected range: {total_p}"

