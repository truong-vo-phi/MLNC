"""
Check that checkpoint selection follows validation accuracy and strictly preserves earlier epoch on ties.
"""

from pathlib import Path

import torch

from src.trainer import Trainer


def test_checkpoint_selects_validation_accuracy_and_preserves_ties(tmp_path: Path):
    model = torch.nn.Linear(1, 2)
    model.num_classes = 2
    config = {
        "training": {
            "epochs": 3,
            "gradient_accumulation": 1,
            "backbone_lr": 1e-3,
            "head_lr": 1e-3,
        },
        "logging": {"metric_for_best": "acc"},
    }

    trainer = Trainer(
        model,
        train_loader=[None],
        val_loader=[None],
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        model_name="selection_test",
    )

    # Simulated validation accuracies: Epoch 1 = 0.5, Epoch 2 = 0.8, Epoch 3 = 0.8 (Tie with Epoch 2)
    eval_accs = iter([0.5, 0.8, 0.8])
    trainer.train_epoch = lambda epoch: (0.1, 0.5)

    def mock_eval(_loader):
        acc = next(eval_accs)
        metrics = {
            "acc": acc,
            "precision": acc,
            "recall": acc,
            "macro_f1": acc,
            "micro_f1": acc,
            "mAP": acc,
        }
        return 0.2, metrics, {}

    trainer.evaluate = mock_eval
    result = trainer.fit()

    # Must preserve Epoch 2 (earlier epoch on tie with Epoch 3)
    assert result["best_epoch"] == 2
    assert result["best_metric"] == 0.8

    state = torch.load(tmp_path / "selection_test" / "best_model.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 2
    assert state["best_metric"] == 0.8


def test_early_stopping_patience_5(tmp_path: Path):
    model = torch.nn.Linear(1, 2)
    model.num_classes = 2
    config = {
        "training": {
            "epochs": 15,
            "gradient_accumulation": 1,
            "backbone_lr": 1e-3,
            "head_lr": 1e-3,
            "early_stopping": True,
            "patience": 5,
        },
        "logging": {"metric_for_best": "acc"},
    }

    trainer = Trainer(
        model,
        train_loader=[None],
        val_loader=[None],
        config=config,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        model_name="early_stop_test",
    )

    # Best at Epoch 2 (0.75), then 5 epochs with no improvement (Epochs 3, 4, 5, 6, 7) -> stops at Epoch 7
    eval_accs = iter([0.50, 0.75, 0.74, 0.75, 0.70, 0.72, 0.73, 0.90, 0.95])
    trainer.train_epoch = lambda epoch: (0.1, 0.5)

    def mock_eval(_loader):
        acc = next(eval_accs)
        return 0.2, {"acc": acc, "precision": acc, "recall": acc, "macro_f1": acc, "micro_f1": acc, "mAP": acc}, {}

    trainer.evaluate = mock_eval
    result = trainer.fit()

    assert result["stopped_early"] is True
    assert result["best_epoch"] == 2
    assert result["best_metric"] == 0.75
    assert len(result["history"]) == 7
