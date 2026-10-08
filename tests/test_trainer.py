import logging

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.core import BaseModel
from src.train import StepOutput, TrainConfig, Trainer


class Regressor(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    @classmethod
    def from_yaml(cls, path):
        return cls()

    def convert_checkpoint(self, checkpoint):
        return checkpoint["model"]

    def forward(self, x):
        return self.linear(x)


class ToyTrainer(Trainer):
    selection_metric = "score"

    def __init__(self, *args, scores, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = iter(scores)
        self.began = 0

    def configure_optimizer(self, steps_per_epoch):
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    def begin_epoch(self):
        self.began += 1

    def training_step(self, batch):
        x, y = batch
        return StepOutput(loss=((self.model(x) - y) ** 2).mean(), size=len(x))

    def evaluate(self, loader, desc="evaluate"):
        return {"score": next(self.scores)}


def make_loader():
    x = torch.linspace(-1, 1, 8).unsqueeze(1)
    return DataLoader(list(zip(x, 2 * x, strict=True)), batch_size=4)


def make_trainer(tmp_path, scores, epochs=10, patience=2):
    config = TrainConfig(epochs=epochs, patience=patience)
    return ToyTrainer(
        Regressor(), config, torch.device("cpu"), logging.getLogger("test"), "toy", tmp_path / "best.pt", scores=scores
    )


def test_fit_tracks_best_and_restores(tmp_path):
    trainer = make_trainer(tmp_path, [0.1, 0.5, 0.3, 0.2], epochs=4, patience=5)
    result = trainer.fit(make_loader(), make_loader())
    assert result.best_epoch == 2
    assert result.best_score == 0.5
    assert result.epochs_run == 4
    assert [row["epoch"] for row in result.history] == [1, 2, 3, 4]
    assert trainer.began == 4
    assert (tmp_path / "best.pt").is_file()


def test_fit_early_stops(tmp_path):
    trainer = make_trainer(tmp_path, [0.5, 0.4, 0.3, 0.2, 0.1], epochs=10, patience=2)
    result = trainer.fit(make_loader(), make_loader())
    assert result.epochs_run == 3
    assert result.best_epoch == 1
    assert result.to_dict()["selection_metric"] == "score"


def test_on_checkpoint_fires_on_each_new_best(tmp_path):
    calls = []
    trainer = make_trainer(tmp_path, [0.1, 0.5, 0.3, 0.6], epochs=4, patience=5)
    trainer.on_checkpoint = lambda path, epoch, score: calls.append((path.name, epoch, score))
    trainer.fit(make_loader(), make_loader())
    assert calls == [("best.pt", 1, 0.1), ("best.pt", 2, 0.5), ("best.pt", 4, 0.6)]
