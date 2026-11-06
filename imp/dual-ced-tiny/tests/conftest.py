"""
Pytest configuration and shared fixtures for Dual-CED-Tiny.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def sample_wave(device: torch.device) -> torch.Tensor:
    return torch.randn(2, 64000, device=device) * 0.05


@pytest.fixture
def sample_spec(device: torch.device) -> torch.Tensor:
    return torch.randn(2, 1, 64, 398, device=device)
