from __future__ import annotations

import copy
import logging
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.loss.frcrn import si_snr
from src.metrics import SnrBucketMeter
from src.model.frcrn import FRCRNSeparator
from src.utils.batch_size import largest_fitting_batch

from .base import StepOutput, Trainer, TrainingResult
from .config import FRCRNTrainConfig


class FRCRNTrainer(Trainer):
    selection_metric: str = "noise_si_snr"

    def __init__(
        self,
        model: FRCRNSeparator,
        config: FRCRNTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
        epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path, on_checkpoint)
        self.selection_metric = config.selection_metric
        self.resume_path = Path(resume_path) if resume_path else None
        self.drive_backup = drive_backup
        self.epoch_callback = epoch_callback
        self._last_scores: dict[str, float] = {}
        self._last_train: dict[str, float] = {}
        self._epoch_records: list[dict[str, float]] = []
        self._past_history: list[dict[str, float]] = []
        self._completed_epochs = 0
        self._current_epoch = 0
        self._resume_state: dict[str, Any] | None = None
        self._resumed_best_state: dict[str, Tensor] | None = None

    @property
    def checkpoint_module(self) -> nn.Module:
        return self.model.network

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        self.optimizer = Adam(self.trainable_parameters, lr=self.config.lr)
        if self._resume_state is not None:
            self.optimizer.load_state_dict(self._resume_state["optimizer"])
            self._resume_state = None
            self.logger.info("resume: optimizer khôi phục từ epoch %d (lr %.1e)", self._completed_epochs, self.lr)

    def save_resume(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.resume_path
        if self.optimizer is None or self._current_epoch == 0 or path is None:
            raise ValueError("optimizer, a finished epoch and a resume path are required for save_resume")
        payload = {
            "epoch": self._current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self._best_score,
            "best_epoch": self._best_epoch,
            "best_state": copy.deepcopy(self._best_state) if self._best_state is not None else None,
            "epoch_records": list(self._epoch_records),
        }
        partial = path.with_name(path.name + ".part")
        torch.save(payload, partial)
        os.replace(partial, path)
        return path

    def load_resume(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(data["model"])
        self._completed_epochs = int(data["epoch"])
        self._best_score = float(data.get("best_score", -float("inf")))
        self._best_epoch = int(data.get("best_epoch", 0))
        self._resumed_best_state = data.get("best_state")
        self._resume_state = data
        self._past_history = list(data.get("epoch_records", []))
        self._epoch_records = list(self._past_history)
        self.logger.info(
            "resume: %s từ epoch %d (best %.4f @ %d)",
            self.name,
            self._completed_epochs,
            self._best_score,
            self._best_epoch,
        )

    def snapshot(self) -> dict[str, Tensor]:
        if self._resumed_best_state is not None:
            state, self._resumed_best_state = copy.deepcopy(self._resumed_best_state), None
            return state
        return super().snapshot()

    def skip_epoch(self, epoch: int) -> bool:
        return epoch <= self._completed_epochs

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        result = super().fit(train_loader, val_loader)
        result.history = self._past_history + result.history
        return result

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        if epoch <= self._completed_epochs:
            return {}
        self._current_epoch = epoch
        if self.config.lr_decay != 1.0 and epoch > 1:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.config.lr_decay
        self._last_train = super().run_epoch(loader, epoch)
        return self._last_train

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        mixture, noise = batch["waveform"].to(self.device), batch["noise"].to(self.device)
        return StepOutput(loss=self.model.loss(mixture, noise), size=len(mixture))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        total_loss, count = 0.0, 0
        noise_meter = SnrBucketMeter(self.config.high_snr_levels)
        speech = SnrBucketMeter(self.config.high_snr_levels)
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            mixture, noise = batch["waveform"].to(self.device), batch["noise"].to(self.device)
            size = len(mixture)
            estimate = self.model(mixture)
            total_loss += self.model.loss(mixture, noise).item() * size
            noise_meter.update(si_snr(noise, estimate), batch["snr_db"])
            speech.update(si_snr(mixture - noise, mixture - estimate), batch["snr_db"])
            count += size
        scores = {
            "loss": total_loss / count,
            **noise_meter.compute("noise_si_snr"),
            **speech.compute("speech_si_snr"),
        }
        self._last_scores = scores
        return scores

    def end_epoch(self, epoch: int) -> None:
        self._epoch_records.append(
            {"epoch": epoch, **self._last_train, **{f"val_{k}": v for k, v in self._last_scores.items()}}
        )
        if self.epoch_callback is not None and self._last_scores:
            self.epoch_callback(epoch, {**self._last_scores, **self._last_train, "lr": self.lr})
        if self.resume_path is None:
            return
        self.save_resume(self.resume_path)
        if self.drive_backup is not None:
            self.drive_backup(self.resume_path)

    def baseline(self, loader: DataLoader) -> float | None:
        if self._completed_epochs > 0:
            return None
        scores = self.evaluate(loader, desc=f"{self.name} - baseline")
        self.logger.info(
            "%s baseline (pretrained) val loss %.3f | noise si-snr %.2f dB | %s %.2f dB",
            self.name,
            scores["loss"],
            scores["noise_si_snr"],
            self.selection_metric,
            scores[self.selection_metric],
        )
        return scores[self.selection_metric]

    def on_no_improvement(self, epoch: int, stale: int) -> None:
        if stale != self.config.halve_lr_after:
            return
        self.restore_best()
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.5
        self.logger.info("%s reload best epoch %d, lr -> %.1e", self.name, self._best_epoch, self.lr)


def scale_batch_size(
    config: FRCRNTrainConfig,
    model_yaml: str | Path,
    device: torch.device,
    sample_rate: int,
    logger: logging.Logger,
) -> FRCRNTrainConfig:
    probe = FRCRNSeparator.from_yaml(model_yaml).to(device).train()
    samples = int(config.segment_seconds * sample_rate)

    def step(size: int) -> None:
        mixture = torch.randn(size, samples, device=device) * 0.1
        noise = torch.randn(size, samples, device=device) * 0.05
        probe.loss(mixture, noise).backward()
        probe.zero_grad(set_to_none=True)

    fitted = largest_fitting_batch(step, config.batch_candidates, device, config.vram_fraction)
    if fitted is None:
        logger.info("batch size: keep %d (no probe or no candidate fits)", config.batch_size)
        return config
    logger.info(
        "batch size: %d -> %d (largest of %s within %.0f%% VRAM)",
        config.batch_size,
        fitted,
        list(config.batch_candidates),
        config.vram_fraction * 100,
    )
    return replace(config, batch_size=fitted)
