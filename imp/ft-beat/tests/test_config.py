from __future__ import annotations

import pytest
from ft_beat.config import (
    PACKAGE_ROOT,
    FTBeatConfig,
    FTBeatDataConfig,
    FTBeatFRCRNConfig,
    FTBeatTrainConfig,
    load_config,
    smoke_config,
)


def test_load_default_config() -> None:
    cfg_path = PACKAGE_ROOT / "config" / "ft_beat.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg, FTBeatConfig)
    assert cfg.train.epochs == 10
    assert cfg.train.batch_size == 16
    assert cfg.train.accumulation_steps == 2
    assert cfg.frcrn.target_rms > 0
    assert cfg.data.dataset_key == "mixture-36l"


def test_smoke_config() -> None:
    cfg_path = PACKAGE_ROOT / "config" / "ft_beat.yaml"
    cfg = load_config(cfg_path)
    smoke = smoke_config(cfg)
    assert smoke.train.epochs == 1
    assert smoke.train.batch_size == 4
    assert smoke.data.num_workers == 0
    assert smoke.data.max_train_samples == 32


def test_invalid_target_rms() -> None:
    with pytest.raises(ValueError, match="target_rms must be positive"):
        FTBeatFRCRNConfig(target_rms=-1.0)


def test_invalid_accumulation_steps() -> None:
    with pytest.raises(ValueError, match="accumulation_steps must be positive"):
        FTBeatTrainConfig(accumulation_steps=0)


def test_invalid_data_duration() -> None:
    with pytest.raises(ValueError, match="duration must be positive"):
        FTBeatDataConfig(duration=-2.0)
