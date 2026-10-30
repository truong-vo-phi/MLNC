from __future__ import annotations

import pytest

from ema_beat.config import (
    PACKAGE_ROOT,
    EMAConfig,
    EMABeatConfig,
    EMABeatDataConfig,
    EMABeatTrainConfig,
    load_config,
    smoke_config,
)


def test_ema_config_defaults() -> None:
    cfg = EMAConfig()
    assert cfg.decay == 0.9995
    assert cfg.warmup_steps == 1000
    assert cfg.warmup_decay_start == 0.90
    assert cfg.selection_target == "ema"
    assert cfg.temperature == 4.0
    assert cfg.crd_weight == 0.8
    assert cfg.crd_dim == 128
    assert cfg.crd_negatives == 4096
    assert cfg.crd_temperature == 0.07


def test_ema_config_validation() -> None:
    with pytest.raises(ValueError, match="decay must be in"):
        EMAConfig(decay=1.5)

    with pytest.raises(ValueError, match="decay must be in"):
        EMAConfig(decay=0.0)

    with pytest.raises(ValueError, match="warmup_decay_start"):
        EMAConfig(decay=0.95, warmup_decay_start=0.98)

    with pytest.raises(ValueError, match="selection_target"):
        EMAConfig(selection_target="invalid")

    with pytest.raises(ValueError, match="crd_weight"):
        EMAConfig(crd_weight=-0.1)

    with pytest.raises(ValueError, match="crd_dim"):
        EMAConfig(crd_dim=0)

    with pytest.raises(ValueError, match="crd_negatives"):
        EMAConfig(crd_negatives=0)

    with pytest.raises(ValueError, match="crd_temperature"):
        EMAConfig(crd_temperature=-0.05)


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="lr must be positive"):
        EMABeatTrainConfig(lr=-0.01)

    with pytest.raises(ValueError, match="betas must be a pair"):
        EMABeatTrainConfig(betas=(0.9, 1.2))

    with pytest.raises(ValueError, match="selection_metric"):
        EMABeatTrainConfig(selection_metric="non_existent")


def test_load_yaml_config() -> None:
    yaml_path = PACKAGE_ROOT / "config" / "ema_beat.yaml"
    cfg = load_config(yaml_path)
    assert isinstance(cfg, EMABeatConfig)
    assert cfg.ema.decay == 0.9995
    assert cfg.train.selection_metric == "macro_f1"


def test_smoke_config() -> None:
    cfg = load_config(PACKAGE_ROOT / "config" / "ema_beat.yaml")
    smoke = smoke_config(cfg)
    assert smoke.train.epochs == 1
    assert smoke.train.batch_size == 8
    assert smoke.data.max_train_samples == 64
