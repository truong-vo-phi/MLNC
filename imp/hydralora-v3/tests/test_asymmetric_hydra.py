import sys
from pathlib import Path
HYDRALORA_DIR = Path(__file__).resolve().parent.parent
for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]
if str(HYDRALORA_DIR) not in sys.path:
    sys.path.insert(0, str(HYDRALORA_DIR))

from unittest.mock import patch
import pytest
import torch
import torch.nn as nn

from src.models.moe_hydra import AsymmetricHydraLinear, CEDHydraLoRAV3


def test_asymmetric_hydra_linear_forward():
    base = nn.Linear(192, 768)
    layer = AsymmetricHydraLinear(base, r_A=24, r_low=8, r_high=24)
    x = torch.randn(4, 97, 192)

    # 1. Forward without alpha (should be equal to base)
    out_no_alpha = layer(x)
    assert out_no_alpha.shape == (4, 97, 768)
    assert torch.allclose(out_no_alpha, base(x), atol=1e-5)

    # 2. Forward with alpha = 0 (low expert only)
    alpha_0 = torch.zeros(4, 1)
    layer.set_alpha(alpha_0)
    out_alpha_0 = layer(x)
    assert out_alpha_0.shape == (4, 97, 768)

    # 3. Forward with alpha = 1 (high expert only)
    alpha_1 = torch.ones(4, 1)
    layer.set_alpha(alpha_1)
    out_alpha_1 = layer(x)
    assert out_alpha_1.shape == (4, 97, 768)


@patch("src.models.moe_hydra.load_ced_tiny", return_value=None)
def test_ced_hydralora_v3_parameter_budget(mock_load):
    m = CEDHydraLoRAV3(target_blocks=(4, 5, 6, 7, 8, 9, 10, 11), r_A=24, r_low=8, r_high=24)
    total_params = sum(p.numel() for p in m.parameters())
    params_m = total_params / 1e6

    print(f"Total parameters: {params_m:.3f}M")
    assert params_m < 6.0, f"Parameter count {params_m:.3f}M exceeds 6.0M constraint!"
    assert params_m > 5.5, f"Parameter count {params_m:.3f}M is unexpectedly low!"


@patch("src.models.moe_hydra.load_ced_tiny", return_value=None)
def test_ced_hydralora_v3_shapes(mock_load):
    m = CEDHydraLoRAV3(target_blocks=(4, 5, 6, 7, 8, 9, 10, 11), r_A=24, r_low=8, r_high=24)
    dummy_audio = torch.randn(2, 64000)

    out = m(dummy_audio, return_dict=True)
    assert "logits" in out
    assert "logits_low" in out
    assert "logits_high" in out
    assert "alpha" in out

    assert out["logits"].shape == (2, 36)
    assert out["logits_low"].shape == (2, 36)
    assert out["logits_high"].shape == (2, 36)
    assert out["alpha"].shape == (2, 1)
