import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.loss import KDMoELossConfig
from src.model.kdmoe import KDMoEConfig, KDMoEModel
from src.train import KDMoETrainConfig, KDMoETrainer
from src.utils.seed import set_seed

NUM_CLASSES = 4
SNRS = (-5, 0, 5, 10, 15, 20)


def make_model() -> KDMoEModel:
    set_seed(0)
    config = KDMoEConfig(
        d=32,
        heads=2,
        n_hybrid=1,
        n_vit=1,
        hybrid_gru_hidden=16,
        final_gru_hidden=8,
        expert_hidden=16,
        dropout=0.0,
        num_classes=NUM_CLASSES,
        num_groups=2,
    )
    return KDMoEModel(config)


def make_membership() -> torch.Tensor:
    membership = torch.zeros(NUM_CLASSES, 2)
    membership[: NUM_CLASSES // 2, 0] = 1.0
    membership[NUM_CLASSES // 2 :, 1] = 1.0
    return membership


def make_item(index: int) -> dict:
    label = index % NUM_CLASSES
    return {
        "waveform": torch.randn(64000),
        "noise": torch.randn(64000),
        "label": torch.tensor(label),
        "group": torch.tensor(0 if label < NUM_CLASSES // 2 else 1),
        "snr_db": torch.tensor(float(SNRS[index % len(SNRS)])),
        "teacher_logits36": torch.randn(NUM_CLASSES),
        "noise_ytid": f"clip{index // 2}",
    }


def make_loader(samples: int = 24) -> DataLoader:
    return DataLoader([make_item(index) for index in range(samples)], batch_size=8)


def make_trainer(tmp_path: Path, stage: int, resume_path: Path | None, **overrides) -> KDMoETrainer:
    config = replace(KDMoETrainConfig(epochs=2, patience=2, stage1_epochs=1, grad_clip=5.0), **overrides)
    return KDMoETrainer(
        make_model(),
        config,
        KDMoELossConfig(),
        make_membership(),
        torch.device("cpu"),
        logging.getLogger("test-kdmoe-trainer"),
        f"stage{stage}",
        tmp_path / f"best_stage{stage}.pt",
        stage=stage,
        resume_path=resume_path,
    )


def test_stage1_then_stage2_fit_smoke(tmp_path):
    loader = make_loader()
    resume1 = tmp_path / "resume_s1.pt"
    trainer1 = make_trainer(tmp_path, stage=1, resume_path=resume1)
    result1 = trainer1.fit(loader, loader)
    assert result1.epochs_run >= 1
    assert resume1.is_file()

    resume2 = tmp_path / "resume_s2.pt"
    trainer2 = make_trainer(tmp_path, stage=2, resume_path=resume2)
    result2 = trainer2.fit(loader, loader)
    assert result2.epochs_run >= 1
    assert resume2.is_file()
    payload = torch.load(resume2, map_location="cpu", weights_only=False)
    assert payload["stage"] == 2
    assert payload["epoch"] >= 1
    assert len(payload["history"]) >= 1


def test_resume_restores_state_and_skips_done_epochs(tmp_path):
    loader = make_loader()
    resume = tmp_path / "resume.pt"
    first = make_trainer(tmp_path, stage=2, resume_path=resume)
    first.fit(loader, loader)
    first_payload = torch.load(resume, map_location="cpu", weights_only=False)

    second = make_trainer(tmp_path, stage=2, resume_path=resume)
    second.load_resume(resume)
    assert second._completed_epochs == first_payload["epoch"]
    assert second._best_score == pytest.approx(first_payload["best_score"])
    assert len(second.history) == len(first_payload["history"])

    second.configure_optimizer(steps_per_epoch=3)
    assert second.optimizer is not None
    assert second.scheduler is not None
    for name, value in first_payload["ema"].items():
        assert torch.allclose(second.ema.shadow[name].cpu(), value.cpu())

    assert second.run_epoch(loader, 1) == {}
    with torch.no_grad():
        weights_before = second.model.state_dict()["patch_embed.proj.weight"].clone()
    assert second.run_epoch(loader, 1) == {}
    assert torch.allclose(second.model.state_dict()["patch_embed.proj.weight"], weights_before)


def test_resume_restores_best_weights_on_snapshot(tmp_path):
    loader = make_loader()
    resume = tmp_path / "resume.pt"
    first = make_trainer(tmp_path, stage=2, resume_path=resume)
    first.fit(loader, loader)
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    assert payload["best_state"] is not None

    second = make_trainer(tmp_path, stage=2, resume_path=resume)
    second.load_resume(resume)
    state = second.snapshot()
    reference = payload["best_state"]
    assert set(state) == set(reference)
    assert torch.allclose(state["patch_embed.proj.weight"], reference["patch_embed.proj.weight"])


def test_load_resume_rejects_stage_mismatch(tmp_path):
    loader = make_loader()
    resume = tmp_path / "resume.pt"
    first = make_trainer(tmp_path, stage=1, resume_path=resume)
    first.fit(loader, loader)

    second = make_trainer(tmp_path, stage=2, resume_path=resume)
    with pytest.raises(ValueError, match="stage"):
        second.load_resume(resume)


def test_save_resume_requires_training(tmp_path):
    trainer = make_trainer(tmp_path, stage=2, resume_path=tmp_path / "resume.pt")
    with pytest.raises(ValueError, match="configure_optimizer"):
        trainer.save_resume()
