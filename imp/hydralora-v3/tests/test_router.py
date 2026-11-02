import sys
from pathlib import Path
HYDRALORA_DIR = Path(__file__).resolve().parent.parent
for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]
if str(HYDRALORA_DIR) not in sys.path:
    sys.path.insert(0, str(HYDRALORA_DIR))

import pytest
import torch

from src.router import SelfContainedSpectralTokenRouter


def test_router_shapes_and_bounds():
    router = SelfContainedSpectralTokenRouter(
        spectral_dim=6,
        token_dim=192,
        token_proj_dim=16,
        hidden_dim=32,
    )

    spectral_feat = torch.randn(8, 6)
    mid_token = torch.randn(8, 192)

    alpha, gates = router(spectral_feat, mid_token)

    assert alpha.shape == (8, 1)
    assert gates.shape == (8, 2)

    # Check bounds
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)
    assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_router_target_soft_label():
    router = SelfContainedSpectralTokenRouter()
    snr_low = torch.tensor([-5.0, 0.0])
    snr_mid = torch.tensor([15.0])
    snr_high = torch.tensor([20.0])

    t_low = router.compute_target_soft_label(snr_low)
    t_mid = router.compute_target_soft_label(snr_mid)
    t_high = router.compute_target_soft_label(snr_high)

    assert torch.all(t_low < 0.1)
    assert torch.allclose(t_mid, torch.tensor([[0.5]]), atol=0.01)
    assert torch.all(t_high > 0.8)
