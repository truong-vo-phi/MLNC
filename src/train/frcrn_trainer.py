from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.loss.frcrn import si_snr
from src.model.frcrn import FRCRNSeparator

from .base import StepOutput, Trainer
from .config import FRCRNTrainConfig


class FRCRNTrainer(Trainer):
    selection_metric: ClassVar[str] = "noise_si_snr"

    def __init__(
        self,
        model: FRCRNSeparator,
        config: FRCRNTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path, on_checkpoint)

    @property
    def checkpoint_module(self) -> nn.Module:
        return self.model.network

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        self.optimizer = Adam(self.trainable_parameters, lr=self.config.lr)

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        mixture, noise = batch["waveform"].to(self.device), batch["noise"].to(self.device)
        return StepOutput(loss=self.model.loss(mixture, noise), size=len(mixture))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        total_loss, total_si_snr, count = 0.0, 0.0, 0
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            mixture, noise = batch["waveform"].to(self.device), batch["noise"].to(self.device)
            size = len(mixture)
            total_loss += self.model.loss(mixture, noise).item() * size
            total_si_snr += si_snr(noise, self.model(mixture)).sum().item()
            count += size
        return {"loss": total_loss / count, "noise_si_snr": total_si_snr / count}

    def baseline(self, loader: DataLoader) -> float:
        scores = self.evaluate(loader, desc=f"{self.name} - baseline")
        self.logger.info(
            "%s baseline (pretrained) val loss %.3f | noise si-snr %.2f dB",
            self.name,
            scores["loss"],
            scores["noise_si_snr"],
        )
        return scores["noise_si_snr"]

    def on_no_improvement(self, epoch: int, stale: int) -> None:
        if stale != self.config.halve_lr_after:
            return
        self.restore_best()
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.5
        self.logger.info("%s reload best epoch %d, lr -> %.1e", self.name, self._best_epoch, self.lr)
