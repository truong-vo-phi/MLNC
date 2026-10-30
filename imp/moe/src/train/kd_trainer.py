from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.loss import LogitDistillationLoss, SoftTargetCrossEntropy
from src.metrics import snr_scores
from src.model.student import CEDStudent
from src.utils.files import rms_normalize

from .base import StepOutput, Trainer
from .config import KDLossConfig, KDTrainConfig


class KDTrainer(Trainer):
    selection_metric = "score"

    def __init__(
        self,
        model: CEDStudent,
        teacher: nn.Module | None,
        config: KDTrainConfig,
        loss_config: KDLossConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        train_state_path: str | Path,
        input_rms: float,
        train_eval_loader: DataLoader | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
        on_epoch: Callable[[int, list[dict[str, float]]], None] | None = None,
        resume: bool = False,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path, on_checkpoint)
        self.teacher = teacher
        self.loss_config = loss_config
        self.train_state_path = Path(train_state_path)
        self.input_rms = input_rms
        self.train_eval_loader = train_eval_loader
        self.on_epoch = on_epoch
        self.resume = resume
        self.selection_metric = config.selection_metric
        self.ce = SoftTargetCrossEntropy(loss_config.label_smoothing)
        self.kd = LogitDistillationLoss(loss_config.temperature)
        self._lr_scale = 1.0
        self._start_epoch = 0
        self._history: list[dict[str, float]] = []
        self._last_train_acc: float | None = None
        self._correct = 0.0
        self._seen = 0
        self._kd_sum = 0.0

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        self.optimizer = AdamW(
            self.trainable_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )
        warmup = max(self.config.warmup_steps, 1)
        total_steps = self.config.epochs * steps_per_epoch
        min_ratio = max(self.config.min_lr / self.config.lr, 0.0)

        if getattr(self.config, "lr_schedule", "cosine") == "cosine":
            def schedule(step: int) -> float:
                if step < warmup:
                    return (step + 1) / warmup
                progress = (step - warmup) / max(total_steps - warmup, 1)
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            def schedule(step: int) -> float:
                return self._lr_scale * min(1.0, (step + 1) / warmup)

        self.scheduler = LambdaLR(self.optimizer, schedule)
        if self.resume:
            self._load_train_state()

    def _autocast(self):
        enabled = self.config.amp and self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def _load_train_state(self) -> None:
        if not self.train_state_path.is_file():
            return
        state = torch.load(self.train_state_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self._best_score = float(state.get("best_score", -float("inf")))
        self._best_epoch = int(state.get("best_epoch", 0))
        self._stale = int(state.get("stale", 0))
        self._lr_scale = float(state.get("lr_scale", 1.0))
        self._start_epoch = int(state.get("epoch", 0))
        self._history = list(state.get("history", []))
        if self.checkpoint_path.is_file():
            best = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self._best_state = best["model"]
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.lr * self._lr_scale

    def _save_train_state(self, epoch: int, history: list[dict[str, float]]) -> None:
        self.train_state_path.parent.mkdir(parents=True, exist_ok=True)
        model = {key: value.detach().cpu() for key, value in self.checkpoint_module.state_dict().items()}
        torch.save(
            {
                "model": model,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_score": self._best_score,
                "best_epoch": self._best_epoch,
                "stale": self._stale,
                "lr_scale": self._lr_scale,
                "epoch": epoch,
                "history": history,
            },
            self.train_state_path,
        )

    def initial_epoch(self) -> int:
        return self._start_epoch

    def initial_history(self) -> list[dict[str, float]]:
        return self._history

    def initial_stale(self) -> int:
        return self._stale

    def begin_epoch(self) -> None:
        if self.teacher is not None:
            self.teacher.eval()
        self._correct, self._seen, self._kd_sum = 0.0, 0, 0.0

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        wave = batch["waveform"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True).float()
        with torch.no_grad():
            noise = rms_normalize(wave, self.input_rms)
        if self.config.mixup > 0 and torch.rand(()).item() < self.config.mixup:
            alpha = self.config.mixup_alpha
            lam = torch.distributions.Beta(alpha, alpha).sample((noise.shape[0], 1)).to(self.device)
            lam = torch.maximum(lam, 1.0 - lam)
            order = torch.randperm(noise.shape[0], device=self.device)
            noise = lam * noise + (1.0 - lam) * noise[order]
            target = lam * target + (1.0 - lam) * target[order]
        with self._autocast():
            logits = self.model(noise).logits.float()
            loss = self.ce(logits, target)
            if self.loss_config.weight > 0 and self.teacher is not None:
                with torch.no_grad(), self._autocast():
                    teacher_logits = self.teacher(noise).logits.float()
                kd = self.kd(logits, teacher_logits)
                loss = loss + self.loss_config.weight * kd
                self._kd_sum += kd.item() * len(target)
        self._correct += (logits.argmax(dim=-1) == target.argmax(dim=-1)).sum().item()
        self._seen += len(target)
        return StepOutput(loss=loss, size=len(target))

    def epoch_summary(self) -> dict[str, float]:
        self._last_train_acc = self._correct / max(self._seen, 1)
        return {"train_acc": self._last_train_acc, "kd": self._kd_sum / max(self._seen, 1)}

    @torch.no_grad()
    def _accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = seen = 0
        for batch in loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            noise = rms_normalize(wave, self.input_rms)
            with self._autocast():
                logits = self.model(noise).logits
            target = batch["target"].to(self.device).argmax(dim=1)
            correct += (logits.argmax(dim=-1) == target).sum().item()
            seen += len(target)
        return correct / max(seen, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        if self.train_eval_loader is not None:
            self._last_train_acc = self._accuracy(self.train_eval_loader)
        self.model.eval()
        probabilities, labels, snrs = [], [], []
        progress = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
        for batch in progress:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            noise = rms_normalize(wave, self.input_rms)
            with self._autocast():
                logits = self.model(noise).logits
            probabilities.append(logits.float().softmax(dim=-1).cpu())
            labels.append(batch["target"].argmax(dim=1))
            snrs.append(batch["snr_db"])
        scores = snr_scores(
            torch.cat(probabilities).numpy(),
            torch.cat(labels).numpy(),
            torch.cat(snrs).numpy(),
            self.model.config.num_classes,
        )
        gap = 0.0 if self._last_train_acc is None else max(0.0, self._last_train_acc - scores["top1_accuracy"])
        scores["gap"] = gap
        scores["score"] = scores["macro_acc"] - self.config.gap_penalty * gap
        return scores

    def on_no_improvement(self, epoch: int, stale: int) -> None:
        if getattr(self.config, "lr_schedule", "cosine") == "cosine":
            return
        if stale % self.config.lr_patience:
            return
        self._lr_scale = max(self._lr_scale * self.config.lr_decay, self.config.min_lr / self.config.lr)
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr * self._lr_scale

    def on_epoch_end(self, epoch: int, history: list[dict[str, float]]) -> None:
        self._save_train_state(epoch, history)
        if self.on_epoch is not None:
            self.on_epoch(epoch, history)
