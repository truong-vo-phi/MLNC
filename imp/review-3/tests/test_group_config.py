"""Guard group selection semantics and effective CE-only model profiles."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_all_models import CANONICAL_MODELS, resolve_model_config
from src.metrics import compute_macro_acc


def test_macro_acc_balances_classes_then_snrs():
    targets = np.tile([0] * 9 + [1], 6)
    snrs = np.repeat([-5, 0, 5, 10, 15, 20], 10)
    probs = np.tile([.9, .1], (60, 1))
    assert np.mean(probs.argmax(1) == targets) == pytest.approx(.9)
    assert compute_macro_acc(probs, targets, snrs) == pytest.approx(.5)
    with pytest.raises(ValueError, match="Missing validation SNR"):
        compute_macro_acc(probs[:50], targets[:50], snrs[:50])


def test_group_profiles_and_user_selected_lr_fallback():
    root = Path(__file__).resolve().parents[1]
    common = yaml.safe_load((root / "config/config.yaml").read_text(encoding="utf-8"))
    assert len(CANONICAL_MODELS) == 9
    for name in CANONICAL_MODELS:
        cfg = resolve_model_config(name, common)
        assert cfg["training"]["epochs"] == 50
        assert cfg["training"]["label_smoothing"] == .1
        assert cfg["ce_only"]["pretrained"] is False
        assert cfg["ce_only"]["stage1_epochs_applied"] == 0
        assert cfg["logging"]["metric_for_best"] == "macro_acc"
    ced = resolve_model_config("ced_small", common)
    assert ced["training"]["lr"] == 1e-4
    assert ced["training"]["layer_decay"] == .3
    assert ced["training"]["drop_path"] == .1
    assert ced["frontend"]["n_mels"] == 64
    assert ced["frontend"]["num_frames"] == 400
    mn10 = resolve_model_config("efficient_mn10", common)
    assert mn10["training"]["lr"] == 6e-5
    assert mn10["frontend"]["specaug_freq_masks"] == 0
    beats = resolve_model_config("beats", common)
    assert beats["training"]["lr"] == 1e-5
    assert beats["training"]["head_lr"] == 1e-4
