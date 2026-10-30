from __future__ import annotations

from pathlib import Path

import pytest

from src.model.student import LogMelConfig, StudentConfig
from src.train import DatasetConfig, KDLossConfig, KDTrainConfig

CONFIG = Path(__file__).resolve().parents[1] / "config" / "kd_beat_ced.yaml"


def test_yaml_sections_parse() -> None:
    dataset = DatasetConfig.from_yaml(CONFIG)
    student = StudentConfig.from_yaml(CONFIG)
    train = KDTrainConfig.from_yaml(CONFIG)
    loss = KDLossConfig.from_yaml(CONFIG)
    frontend = LogMelConfig.from_yaml(CONFIG)
    assert dataset.name == "mixture-36l"
    assert student.freq_bins == 64
    assert student.init == "ced-tiny"
    assert train.epochs == 30
    assert train.selection_metric == "score"
    assert loss.temperature == 4.0
    assert frontend.n_mels == 64


def test_student_config_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        StudentConfig(freq_bins=30)
    with pytest.raises(ValueError):
        StudentConfig(init="unknown")
    with pytest.raises(ValueError):
        StudentConfig(bias_init="teacher")


def test_train_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        KDTrainConfig(mixup=1.5)
    with pytest.raises(ValueError):
        KDTrainConfig(lr_patience=0)
    with pytest.raises(ValueError):
        KDLossConfig(temperature=0.0)
    with pytest.raises(ValueError):
        KDLossConfig(label_smoothing=1.0)
