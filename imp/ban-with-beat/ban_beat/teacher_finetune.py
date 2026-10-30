from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.load_pretrain import download_pretrained
from src.model.beat import BEATClassifier
from src.model.frcrn import FRCRNSeparator
from src.train import BEATTrainConfig, BEATTrainer, TrainingResult
from src.utils.paths import RunPaths

from .config import BANConfig, resolve_path
from .teacher import DEFAULT_INPUT_RMS, BANTeacher, TeacherMetadata


def train_teacher(
    classifier: BEATClassifier,
    separator: nn.Module | None,
    train_config: BEATTrainConfig,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path,
    resume_path: Path,
) -> TrainingResult:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = BEATTrainer(
        classifier,
        train_config,
        device,
        logger,
        "teacher",
        checkpoint_path,
        separator=separator,
        resume_path=resume_path,
    )
    if resume_path.is_file():
        trainer.load_resume(resume_path)
    return trainer.fit(loaders["train"], loaders["validation"])


def restore_teacher_classifier(classifier: BEATClassifier, checkpoint_path: Path) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(payload["model"])
    return int(payload.get("epoch", 0))


def finetune_teacher(
    config: BANConfig,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run: RunPaths,
) -> tuple[BANTeacher, TeacherMetadata, Path]:
    separator: nn.Module | None = None
    if config.data.nhat_cache_dir is None:
        frcrn = FRCRNSeparator.from_yaml(resolve_path(config.teacher.frcrn_config))
        frcrn.load_pretrained(resolve_path(config.teacher.frcrn_checkpoint))
        separator = frcrn
    classifier = BEATClassifier.from_yaml(config.model_config)
    checkpoint_path = run.checkpoint_path("teacher")
    if config.teacher.reuse and checkpoint_path.is_file():
        epoch = restore_teacher_classifier(classifier, checkpoint_path)
        logger.info("teacher: nạp %s (epoch %d), bỏ qua huấn luyện", checkpoint_path, epoch)
    else:
        classifier.load_pretrained(download_pretrained("beats"))
        resume_path = run.checkpoint_dir / "resume" / "teacher.pt"
        epoch = train_teacher(
            classifier, separator, config.train, loaders, device, logger, checkpoint_path, resume_path
        ).best_epoch
    input_rms = config.teacher.input_rms if config.teacher.input_rms is not None else DEFAULT_INPUT_RMS
    teacher = BANTeacher(classifier, separator, input_rms).to(device).freeze()
    metadata = TeacherMetadata(
        path=checkpoint_path,
        format="finetune-frcrn-beats-pretrain",
        input_rms=input_rms,
        uses_frcrn=True,
        epoch=epoch,
    )
    return teacher, metadata, checkpoint_path
