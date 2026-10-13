import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.model.beat import BEATClassifier, BEATConfig
from src.train import BEATTrainConfig, BEATTrainer
from src.utils.seed import set_seed

NUM_CLASSES = 4
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def make_config() -> BEATConfig:
    return BEATConfig(
        embed_dim=32,
        encoder_embed_dim=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_embed_dim=64,
        conv_pos=8,
        conv_pos_groups=2,
        num_classes=NUM_CLASSES,
    )


def make_item(index: int) -> dict:
    label = index % NUM_CLASSES
    target = torch.zeros(NUM_CLASSES)
    target[label] = 1.0
    return {
        "waveform": torch.randn(16000),
        "target": target,
        "snr_db": torch.tensor(SNRS[index % len(SNRS)]),
    }


def make_loader(samples: int = 16) -> DataLoader:
    return DataLoader([make_item(index) for index in range(samples)], batch_size=8)


def make_trainer(tmp_path: Path, resume: Path | None) -> BEATTrainer:
    config = BEATTrainConfig(epochs=2, patience=2, warmup_ratio=0.1, batch_size=8)
    set_seed(0)
    return BEATTrainer(
        BEATClassifier(make_config()),
        config,
        torch.device("cpu"),
        logging.getLogger("test-beat-resume"),
        "beat-ft",
        tmp_path / "best.pt",
        resume_path=resume,
    )


def test_fit_smoke_and_resume_roundtrip(tmp_path):
    resume = tmp_path / "resume.pt"
    first = make_trainer(tmp_path, resume)
    first.fit(make_loader(), make_loader())
    assert resume.is_file()
    assert len(first.history) == 2

    payload = torch.load(resume, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2

    second = make_trainer(tmp_path, resume)
    second.load_resume(resume)
    assert second._completed_epochs == 2
    assert len(second.history) == 2
    assert second._best_score == pytest.approx(float(payload["best_score"]))

    second.configure_optimizer(steps_per_epoch=2)
    assert second.optimizer is not None
    before = second.model.state_dict()["backbone.patch_embed.patch_embedding.weight"].clone()
    assert second.run_epoch(make_loader(), 1) == {}
    assert torch.allclose(second.model.state_dict()["backbone.patch_embed.patch_embedding.weight"], before)


def test_resume_restores_best_weights(tmp_path):
    resume = tmp_path / "resume.pt"
    first = make_trainer(tmp_path, resume)
    first.fit(make_loader(), make_loader())
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    assert payload["best_state"] is not None

    second = make_trainer(tmp_path, resume)
    second.load_resume(resume)
    state = second.snapshot()
    assert torch.allclose(
        state["backbone.patch_embed.patch_embedding.weight"],
        payload["best_state"]["backbone.patch_embed.patch_embedding.weight"],
    )


def test_save_resume_requires_training(tmp_path):
    trainer = make_trainer(tmp_path, tmp_path / "resume.pt")
    with pytest.raises(ValueError, match="configure_optimizer"):
        trainer.save_resume()
