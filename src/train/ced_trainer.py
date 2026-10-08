from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.loss import KDLoss
from src.model.beat import BEATTeacher
from src.model.ced import CEDClassifier

from .base import StepOutput
from .beat_trainer import BEATTrainer
from .config import CEDTrainConfig


class CEDTrainer(BEATTrainer):
    def __init__(
        self,
        model: CEDClassifier,
        teacher: BEATTeacher,
        config: CEDTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path)
        self.teacher = teacher.to(device).freeze()
        self.criterion = KDLoss(config.kd_temperature)

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform, labels = self.prepare(batch)
        with torch.no_grad():
            teacher_logits = self.teacher(waveform).logits
        logits = self.model(waveform).logits
        self._labels, self._logits = labels, logits.detach()
        return StepOutput(loss=self.criterion(logits, teacher_logits), size=len(labels))

    def evaluate_teacher(self, loader: DataLoader, desc: str = "teacher") -> dict[str, float]:
        return self._evaluate_with(self.teacher.predict_proba, loader, desc)
