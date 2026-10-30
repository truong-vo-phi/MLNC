from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.core import BaseModel
from src.train import BEATTrainer, StepOutput

from .config import BANLossConfig, BANTrainConfig
from .evaluate import collect_predictions, metrics_by_snr
from .loss import BornAgainLoss, attention_transfer_loss
from .runtime import autocast_context
from .teacher import BANTeacher


class BANTrainer(BEATTrainer):
    def __init__(
        self,
        model: BaseModel,
        config: BANTrainConfig,
        loss_config: BANLossConfig,
        snr_levels: Sequence[float],
        teacher: BANTeacher | None,
        teacher_logits: Mapping[str, Tensor] | None,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
    ) -> None:
        super().__init__(
            model,
            config,
            device,
            logger,
            name,
            checkpoint_path,
            resume_path=resume_path,
            drive_backup=drive_backup,
            on_checkpoint=on_checkpoint,
        )
        self.criterion = BornAgainLoss(
            kd_weight=loss_config.kd_weight,
            temperature=loss_config.temperature,
            label_smoothing=config.label_smoothing,
            confidence_weighting=loss_config.confidence_weighting,
            confidence_power=loss_config.confidence_power,
        )
        self.teacher = teacher
        self.teacher_logits = dict(teacher_logits) if teacher_logits is not None else None
        self.snr_levels = tuple(float(level) for level in snr_levels)
        self.amp = config.amp
        self.selection_metric = config.selection_metric
        self.attention_weight = loss_config.attention_weight
        self.attention_layers = tuple(loss_config.attention_layers) if loss_config.attention_weight > 0 else ()
        self._batch_teacher_logits: Tensor | None = None
        self._teacher_correct = 0
        self._teacher_seen = 0
        self._missing_teacher_epoch = 0
        self._missing_teacher_total = 0

    def begin_epoch(self) -> None:
        super().begin_epoch()
        self._teacher_correct = 0
        self._teacher_seen = 0
        self._missing_teacher_epoch = 0

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform, labels = self.prepare(batch)
        layers = self.attention_layers
        with autocast_context(self.device, self.amp):
            output = self.model(waveform, layers=layers)
        logits = output.logits.float()
        if layers:
            teacher_logits, teacher_hidden = self._teacher_with_hidden(waveform, layers)
        else:
            teacher_logits, teacher_hidden = self._teacher_logits(batch, waveform), ()
        loss = self.criterion(logits, teacher_logits, labels)
        if layers:
            loss = loss + self.attention_weight / 2 * attention_transfer_loss(output.hidden, teacher_hidden)
        self._labels, self._logits = labels, logits.detach()
        self._batch_teacher_logits = teacher_logits.detach() if teacher_logits is not None else None
        return StepOutput(loss=loss, size=len(labels))

    def _teacher_with_hidden(self, waveform: Tensor, layers: tuple[int, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
        if self.teacher is None:
            raise ValueError("a teacher model is required when attention_weight > 0")
        with autocast_context(self.device, self.amp):
            logits, hidden = self.teacher.forward_with_hidden(waveform, layers)
        return logits.float(), hidden

    def _teacher_logits(self, batch: dict[str, Any], waveform: Tensor) -> Tensor | None:
        if self.criterion.kd_weight == 0.0:
            return None
        if self.teacher_logits is not None:
            sample_ids = [str(sample_id) for sample_id in batch["sample_id"]]
            cached = [self.teacher_logits.get(sample_id) for sample_id in sample_ids]
            missing = [position for position, value in enumerate(cached) if value is None]
            if missing:
                if self.teacher is None:
                    raise KeyError("teacher logits are missing and no teacher model is available")
                with autocast_context(self.device, self.amp):
                    computed = self.teacher(waveform[missing]).float()
                for position, value in zip(missing, computed, strict=True):
                    cached[position] = value.detach()
                self._missing_teacher_epoch += len(missing)
                self._missing_teacher_total += len(missing)
            resolved = [value for value in cached if value is not None]
            if len(resolved) != len(cached):
                raise KeyError("teacher logits could not be resolved for every sample")
            return torch.stack([value.to(device=waveform.device, dtype=torch.float32) for value in resolved])
        if self.teacher is None:
            raise ValueError("teacher logits or a teacher model is required when kd_weight > 0")
        with autocast_context(self.device, self.amp):
            return self.teacher(waveform).float()

    def observe(self, step: StepOutput) -> None:
        super().observe(step)
        if self._batch_teacher_logits is None:
            return
        student_labels = self._logits.argmax(dim=-1)
        teacher_labels = self._batch_teacher_logits.argmax(dim=-1)
        self._teacher_correct += (student_labels == teacher_labels).sum().item()
        self._teacher_seen += len(student_labels)

    def epoch_summary(self) -> dict[str, float]:
        summary = super().epoch_summary()
        if self._teacher_seen:
            summary["teacher_agree"] = self._teacher_correct / self._teacher_seen
        if self._missing_teacher_epoch:
            summary["teacher_missing"] = float(self._missing_teacher_epoch)
        return summary

    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        predictions = collect_predictions(
            lambda waveform: self.model(waveform).logits,
            loader,
            self.device,
            self.amp,
            desc=desc,
        )
        scores = metrics_by_snr(predictions, self.snr_levels)
        if self._in_fit and self._current_epoch > 0:
            self._history.append(
                {"epoch": self._current_epoch, **{f"val_{key}": value for key, value in scores.items()}}
            )
            if self.resume_path is not None:
                self.save_resume(self.resume_path)
                if self.drive_backup is not None:
                    self.drive_backup(self.resume_path)
        return scores
