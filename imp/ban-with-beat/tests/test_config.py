from __future__ import annotations

from pathlib import Path

import pytest
from ban_beat.config import (
    PACKAGE_ROOT,
    BANDataConfig,
    BANGenerationConfig,
    BANLossConfig,
    BANTrainConfig,
    load_config,
    resolve_path,
    smoke_config,
)

CONFIG = PACKAGE_ROOT / "config" / "ban_beat.yaml"


def test_load_config() -> None:
    config = load_config(CONFIG)
    assert config.model_config.name == "beat.yaml"
    assert config.teacher.finetune is True
    assert config.teacher.checkpoint is None
    assert config.teacher.frcrn_checkpoint == "pretrained/FRCRN/best_e037_20261020T203958Z.pt"
    assert config.train.amp == "bf16"
    assert config.train.selection_metric == "macro_f1_snr"
    assert config.generations.count == 2
    assert config.loss.temperature == 2.0


def test_smoke_config_is_small() -> None:
    config = smoke_config(load_config(CONFIG))
    assert config.data.max_train_samples == 64
    assert config.data.num_workers == 0
    assert config.train.epochs == 1
    assert config.generations.count == 1
    assert config.generations.ensemble is False
    assert config.teacher.cache_logits is False


def test_resolve_path_uses_repo_root() -> None:
    path = resolve_path("config/beat.yaml")
    assert path.is_file()
    assert Path(path).name == "beat.yaml"


@pytest.mark.parametrize("count", [0, 6])
def test_generation_count_out_of_range(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        BANGenerationConfig(count=count)


def test_generation_needs_enough_seeds() -> None:
    with pytest.raises(ValueError, match="seeds"):
        BANGenerationConfig(count=3, seeds=(1, 2))


def test_generation_rejects_unknown_init() -> None:
    with pytest.raises(ValueError, match="student_init"):
        BANGenerationConfig(student_init="imagenet")


def test_loss_rejects_bad_weight() -> None:
    with pytest.raises(ValueError, match="kd_weight"):
        BANLossConfig(kd_weight=1.5)


def test_loss_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        BANLossConfig(temperature=0.0)


def test_train_rejects_unknown_amp() -> None:
    with pytest.raises(ValueError, match="amp"):
        BANTrainConfig(amp="fp16")


def test_train_rejects_unknown_selection_metric() -> None:
    with pytest.raises(ValueError, match="selection_metric"):
        BANTrainConfig(selection_metric="acc")


def test_data_rejects_bad_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        BANDataConfig(duration=0.0)
