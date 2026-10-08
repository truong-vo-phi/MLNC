from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import BaseModel
from src.loss import CrossEntropyLoss
from src.metrics import ClassificationMetrics
from src.model.frcrn import FRCRNSeparator

from .base import StepOutput, Trainer
from .config import BEATTrainConfig


def cosine_with_warmup(optimizer: AdamW, total_steps: int, warmup_ratio: float, min_lr_ratio: float = 0.0) -> LambdaLR:
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, schedule)


class BEATTrainer(Trainer):
    selection_metric: ClassVar[str] = "macro_f1"

    def __init__(
        self,
        model: BaseModel,
        config: BEATTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        separator: FRCRNSeparator | None = None,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path)
        self.separator = separator.to(device).freeze() if separator is not None else None
        self.criterion = CrossEntropyLoss(config.label_smoothing)
        self._train_metrics = ClassificationMetrics(num_classes=self.num_classes)
        self._labels: Tensor | None = None
        self._logits: Tensor | None = None
        self._correct = 0
        self._seen = 0

    @property
    def num_classes(self) -> int:
        return self.model.config.num_classes

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        self.optimizer = AdamW(
            self.trainable_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=tuple(self.config.betas),
        )
        total_steps = self.config.epochs * steps_per_epoch
        self.scheduler = cosine_with_warmup(
            self.optimizer, total_steps, self.config.warmup_ratio, self.config.min_lr_ratio
        )

    def prepare(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        waveform = batch["waveform"].to(self.device)
        labels = batch["target"].argmax(dim=1).to(self.device)
        if self.separator is not None:
            with torch.no_grad():
                waveform = self.separator(waveform)
        return waveform, labels

    def begin_epoch(self) -> None:
        self._train_metrics.reset()
        self._correct, self._seen = 0, 0

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform, labels = self.prepare(batch)
        logits = self.model(waveform).logits
        self._labels, self._logits = labels, logits.detach()
        return StepOutput(loss=self.criterion(logits, labels), size=len(labels))

    def observe(self, step: StepOutput) -> None:
        self._train_metrics.update(self._labels, self._logits.softmax(dim=-1))
        self._correct += (self._logits.argmax(dim=-1) == self._labels).sum().item()
        self._seen += step.size

    def epoch_summary(self) -> dict[str, float]:
        return {"acc": self._correct / max(self._seen, 1), "mAP": self._train_metrics.compute().mAP}

    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        return self._evaluate_with(self.model.predict_proba, loader, desc)

    @torch.no_grad()
    def _evaluate_with(
        self, predict_proba: Callable[[Tensor], Tensor], loader: DataLoader, desc: str
    ) -> dict[str, float]:
        metrics = ClassificationMetrics(num_classes=self.num_classes)
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            waveform, labels = self.prepare(batch)
            metrics.update(labels, predict_proba(waveform))
        return metrics.compute().to_dict()
