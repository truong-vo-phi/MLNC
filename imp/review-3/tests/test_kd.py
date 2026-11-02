"""
Unit tests for Task 3 Knowledge Distillation components:
- LogitKDLoss
- CRDLoss
- MGDLoss
- CompositeKDLoss (5 methods: logit, crd, mgd, logit+crd, logit+mgd)
- CEDTinyBeatBiasPostLN model
"""

import pytest
import torch
import torch.nn as nn

from src.kd.losses import CompositeKDLoss, CRDLoss, LogitKDLoss, MGDLoss
from src.models import build_model


def test_logit_kd_loss():
    loss_fn = LogitKDLoss(temperature=2.0)
    s_logits = torch.randn(4, 36, requires_grad=True)
    t_logits = torch.randn(4, 36)

    loss = loss_fn(s_logits, t_logits)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert s_logits.grad is not None
    assert torch.any(s_logits.grad != 0.0)


def test_crd_loss():
    loss_fn = CRDLoss(student_dim=192, teacher_dim=768, embed_dim=128, temperature=0.07)
    s_feat = torch.randn(4, 192, requires_grad=True)
    t_feat = torch.randn(4, 768)

    loss = loss_fn(s_feat, t_feat)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert s_feat.grad is not None
    assert torch.any(s_feat.grad != 0.0)


def test_mgd_loss():
    loss_fn = MGDLoss(student_dim=192, teacher_dim=768, mask_ratio=0.5)
    s_feat = torch.randn(4, 192, requires_grad=True)
    t_feat = torch.randn(4, 768)

    loss = loss_fn(s_feat, t_feat)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert s_feat.grad is not None
    assert torch.any(s_feat.grad != 0.0)


@pytest.mark.parametrize("method", ["logit", "crd", "mgd", "logit+crd", "logit+mgd"])
def test_composite_kd_loss_methods(method: str):
    comp_loss = CompositeKDLoss(
        method=method,
        student_dim=192,
        teacher_dim=768,
        num_classes=36,
        temperature=2.0,
    )
    b = 4
    s_logits = torch.randn(b, 36, requires_grad=True)
    t_logits = torch.randn(b, 36)
    s_feat = torch.randn(b, 192, requires_grad=True)
    t_feat = torch.randn(b, 768)
    targets = torch.randint(0, 36, (b,))

    loss, loss_dict = comp_loss(
        student_logits=s_logits,
        teacher_logits=t_logits,
        student_feat=s_feat,
        teacher_feat=t_feat,
        targets=targets,
    )

    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert "ce" in loss_dict
    assert "total" in loss_dict

    if "logit" in method:
        assert "logit_kd" in loss_dict
    if "crd" in method:
        assert "crd" in loss_dict
    if "mgd" in method:
        assert "mgd" in loss_dict

    loss.backward()
    assert s_logits.grad is not None


def test_ced_tiny_beat_bias_postln():
    model = build_model("ced_tiny_beat_bias_postln", num_classes=36)
    assert model.num_classes == 36
    total_p, train_p = model.count_parameters()
    assert 5.0e6 < total_p < 6.0e6

    x = torch.randn(2, 64000)
    logits, feat = model(x, return_features=True)
    assert logits.shape == (2, 36)
    assert feat.shape == (2, 97, 192)

