from __future__ import annotations

import pytest
from ced_tiny.config import (
    PACKAGE_ROOT,
    CEDDataConfig,
    CEDFRCRNConfig,
    CEDTinyConfig,
    CEDTinyPipelineConfig,
    CEDTrainConfig,
    FrontendConfig,
    SpecAugmentConfig,
    load_config,
    smoke_config,
)


def test_load_default_config() -> None:
    cfg_path = PACKAGE_ROOT / "config" / "ced_tiny.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg, CEDTinyPipelineConfig)
    assert cfg.train.epochs == 15
    assert cfg.train.batch_size == 32
    assert cfg.train.accumulation_steps == 1
    assert cfg.ced_tiny.arch == "deit_tiny_patch16_224"
    assert cfg.ced_tiny.num_classes == 36
    assert cfg.ced_tiny.freq_bins == 64
    assert cfg.ced_tiny.frames == 398
    assert cfg.frcrn.target_rms > 0
    assert cfg.data.dataset_key == "mixture-36l"


def test_smoke_config() -> None:
    cfg_path = PACKAGE_ROOT / "config" / "ced_tiny.yaml"
    cfg = load_config(cfg_path)
    smoke = smoke_config(cfg)
    assert smoke.train.epochs == 1
    assert smoke.train.batch_size == 4
    assert smoke.data.num_workers == 0
    assert smoke.data.max_train_samples == 32
    assert smoke.ced_tiny.init == "none"


def test_invalid_target_rms() -> None:
    with pytest.raises(ValueError, match="target_rms must be positive"):
        CEDFRCRNConfig(target_rms=-1.0)


def test_invalid_ced_config() -> None:
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        CEDTinyConfig(num_classes=1)

    with pytest.raises(ValueError, match="freq_bins must be divisible by 16"):
        CEDTinyConfig(freq_bins=63)


def test_invalid_train_config() -> None:
    with pytest.raises(ValueError, match="lr must be positive"):
        CEDTrainConfig(lr=-0.01)

    with pytest.raises(ValueError, match="label_smoothing must be in"):
        CEDTrainConfig(label_smoothing=1.5)


def test_invalid_frontend_and_data() -> None:
    with pytest.raises(ValueError, match="sample_rate, n_mels and num_frames must be positive"):
        FrontendConfig(sample_rate=-100)

    with pytest.raises(ValueError, match="duration must be positive"):
        CEDDataConfig(duration=-1.0)

    with pytest.raises(ValueError, match="mask counts and widths must be non-negative"):
        SpecAugmentConfig(freq_masks=-1)
