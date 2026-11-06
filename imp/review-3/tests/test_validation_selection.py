"""Check that checkpoint selection follows validation accuracy and preserves ties."""
from pathlib import Path

import torch

from src.trainer import Trainer


def test_checkpoint_selects_validation_accuracy(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 2)
    config = {"training": {"epochs": 3, "patience": 5}, "logging": {"metric_for_best": "acc"}}
    trainer = Trainer(model, [None], [None], config=config, device=torch.device("cpu"),
                      checkpoint_dir=tmp_path, model_name="selection")
    results = iter(((0.4, 0.9), (0.7, 0.2), (0.7, 0.8)))
    trainer.train_epoch = lambda epoch: (0.0, 0.0)

    def evaluate(_loader):
        accuracy, f1 = next(results)
        return 0.0, {"acc": accuracy, "macro_f1": f1, "precision": 0.0,
                     "recall": 0.0, "micro_f1": 0.0, "mAP": 0.0}, {}

    trainer.evaluate = evaluate
    result = trainer.fit()
    state = torch.load(tmp_path / "selection" / "best_model.pt", map_location="cpu", weights_only=False)
    assert result["best_epoch"] == 2
    assert state["epoch"] == 2
    assert state["best_metric"] == 0.7
