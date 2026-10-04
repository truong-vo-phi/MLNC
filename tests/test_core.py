from dataclasses import dataclass, replace

import pytest
import torch
from torch import nn

from src.core import BaseConfig, BaseModel, LoadReport
from src.loss import (
    CRDLoss,
    CrossEntropyLoss,
    DistillationLoss,
    FRCRNLoss,
    KDLoss,
    SeparationLoss,
    SupervisedLoss,
)
from src.model.beat import BEATClassifier
from src.model.frcrn import FRCRNConfig, FRCRNSeparator
from src.train import BEATTrainConfig, BEATTrainer, FRCRNTrainConfig, FRCRNTrainer, Trainer


@dataclass(frozen=True)
class DemoConfig(BaseConfig):
    section = "demo"
    a: int = 1
    b: str = "x"


class ToyModel(BaseModel):
    expected_missing_prefixes = ("head.",)

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.head = nn.Linear(2, 3)

    @classmethod
    def from_yaml(cls, path):
        return cls()

    def convert_checkpoint(self, checkpoint):
        return {f"body.{k}": v for k, v in checkpoint["model"].items()}


def test_base_config_roundtrip(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text("demo:\n  a: 5\n  unknown: 1\n", encoding="utf-8")
    config = DemoConfig.from_yaml(path)
    assert config == DemoConfig(a=5)
    assert config.to_dict() == {"a": 5, "b": "x"}
    with pytest.raises(AttributeError):
        config.a = 2


def test_configs_share_base():
    for config_cls in (FRCRNConfig, BEATTrainConfig, FRCRNTrainConfig):
        assert issubclass(config_cls, BaseConfig)


def test_load_pretrained_reports_and_filters_expected_missing(tmp_path):
    model, donor = ToyModel(), ToyModel()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": donor.body.state_dict()}, path)
    report = model.load_pretrained(path)
    assert report == LoadReport()
    assert torch.equal(model.body.weight, donor.body.weight)


def test_load_pretrained_strict_raises_on_unexpected(tmp_path):
    model = ToyModel()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {"weight": model.body.weight, "bias": model.body.bias, "extra": torch.zeros(1)}}, path)
    with pytest.raises(ValueError):
        model.load_pretrained(path)


def test_freeze_and_num_parameters():
    model = ToyModel().freeze()
    assert not model.training
    assert model.num_parameters(trainable_only=True) == 0
    assert model.num_parameters() == 2 * 2 + 2 + 2 * 3 + 3


def test_models_share_interface():
    for model_cls in (BEATClassifier, FRCRNSeparator):
        assert issubclass(model_cls, BaseModel)
        assert callable(model_cls.from_yaml)


def test_loss_hierarchy():
    assert issubclass(CrossEntropyLoss, SupervisedLoss)
    assert issubclass(KDLoss, DistillationLoss) and issubclass(CRDLoss, DistillationLoss)
    assert issubclass(FRCRNLoss, SeparationLoss)
    names = {loss.name for loss in (CrossEntropyLoss, KDLoss, CRDLoss, FRCRNLoss)}
    assert len(names) == 4


def test_trainers_share_template():
    assert issubclass(BEATTrainer, Trainer) and issubclass(FRCRNTrainer, Trainer)
    assert BEATTrainer.selection_metric != FRCRNTrainer.selection_metric


def test_train_config_replace_keeps_type():
    config = replace(BEATTrainConfig(), epochs=1)
    assert isinstance(config, BEATTrainConfig)
    assert config.epochs == 1
