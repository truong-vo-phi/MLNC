"""
Unit tests for Task 4: CED-SingleMoE-SplitFFN (~6.62M params) and 9 Evaluation Metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.metrics import compute_9_metrics_dict, compute_snr_breakdown
from src.models import build_model
from src.models.ced_moe_split import (
    AcousticFeatureExtractor,
    CEDSingleMoESplitFFN,
    LogMel,
    SelfContainedSpectralTokenRouter,
    SplitMoEFFN,
    rms_normalize,
)


def test_rms_normalize_and_acoustic_features():
    wav = torch.randn(2, 64000) * 0.5
    norm_wav = rms_normalize(wav)
    rms_val = norm_wav.square().mean(dim=-1).sqrt()
    assert torch.allclose(rms_val, torch.tensor([0.07925, 0.07925]), atol=1e-4)

    extractor = AcousticFeatureExtractor()
    feat = extractor(norm_wav)
    assert feat.shape == (2, 6)
    assert torch.all(torch.isfinite(feat))


def test_router_forward_and_loss():
    router = SelfContainedSpectralTokenRouter()
    spec_feat = torch.randn(4, 6)
    token = torch.randn(4, 192)

    alpha, gates = router(spec_feat, token)
    assert alpha.shape == (4, 1)
    assert gates.shape == (4, 2)
    assert torch.all((alpha >= 0.0) & (alpha <= 1.0))
    assert torch.allclose(gates.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Test soft target
    snrs = torch.tensor([-5.0, 0.0, 15.0, 20.0])
    target = router.compute_target_soft_label(snrs)
    assert target.shape == (4, 1)
    # At SNR 15dB, target should be 0.50
    assert pytest.approx(target[2].item(), 1e-3) == 0.50
    # Loss calculation
    loss = router.compute_router_loss(alpha, snrs)
    assert loss.item() >= 0.0


def test_split_moe_ffn_identity_warmstart():
    dim = 192
    hidden = 768
    base_mlp = torch.nn.Sequential(
        torch.nn.Linear(dim, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, dim),
    )
    moe = SplitMoEFFN(base_mlp)

    x = torch.randn(2, 97, dim)
    for alpha_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha = torch.full((2, 1), alpha_val)
        out_moe = moe(x, alpha)
        out_base = base_mlp(x)
        # At init, output must be mathematically identical for all alpha
        assert torch.allclose(out_moe, out_base, atol=1e-5)


def test_ced_single_moe_split_instantiation_and_forward():
    model = build_model("ced_single_moe_split", num_classes=36, load_pretrained=False)
    assert isinstance(model, CEDSingleMoESplitFFN)

    params = model.get_num_params()
    total_m = params["total"] / 1e6
    assert 6.0 <= total_m <= 7.0, f"Expected params ~6.6M, got {total_m:.2f}M"

    # Forward with waveform (B, 64000)
    wav = torch.randn(2, 64000)
    logits = model(wav)
    assert logits.shape == (2, 36)

    # Forward return_dict
    d = model(wav, return_dict=True)
    assert "logits" in d
    assert "alpha" in d
    assert "spectral_feat" in d
    assert "block9_token" in d
    assert d["alpha"].shape == (2, 1)
    assert d["spectral_feat"].shape == (2, 6)

    # Forward return_features
    logits_feat, cls_feat = model(wav, return_features=True)
    assert logits_feat.shape == (2, 36)
    assert cls_feat.shape == (2, 192)

    # Forward with spectrogram (B, 1, 64, 398)
    spec = torch.randn(2, 1, 64, 398)
    spec_logits = model(spec)
    assert spec_logits.shape == (2, 36)


def test_compute_9_metrics():
    num_classes = 5
    targets = np.array([0, 1, 2, 3, 4])
    logits = np.eye(num_classes) * 10.0
    alphas = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    m = compute_9_metrics_dict(logits, targets, alphas=alphas, num_classes=num_classes)

    assert "top1" in m
    assert "top3" in m
    assert "balanced_acc" in m
    assert "map" in m
    assert "auc" in m
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    assert "mean_alpha" in m

    assert pytest.approx(m["top1"], 1e-4) == 1.0
    assert pytest.approx(m["top3"], 1e-4) == 1.0
    assert pytest.approx(m["balanced_acc"], 1e-4) == 1.0
    assert pytest.approx(m["mean_alpha"], 1e-4) == 0.60

    # Test SNR breakdown with 9 metrics
    snrs = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
    breakdown = compute_snr_breakdown(
        logits, targets, snrs, alphas=alphas, num_classes=num_classes, use_9_metrics=True
    )
    assert "all" in breakdown
    assert "-5dB" in breakdown
    assert "top1" in breakdown["-5dB"]
    assert "mean_alpha" in breakdown["-5dB"]

