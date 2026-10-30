from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import BaseModel
from src.metrics import ClassificationMetrics
from src.model.frcrn import FRCRNSeparator
from src.train import BEATTrainer, StepOutput
from src.train.beat_trainer import cosine_with_warmup

from .config import FTBeatTrainConfig
from .model import rms_normalize
from .runtime import autocast_context


def get_layer_decay_params(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> list[dict[str, Any]]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    layers = model.encoder.layers
    num_layers = len(layers)
    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    param_groups: dict[float, list[nn.Parameter]] = {}
    no_decay_groups: dict[float, list[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = param.ndim <= 1 or name.endswith(".bias")

        if name.startswith("encoder.layers."):
            layer_idx = int(name.split(".")[2])
            scale = scales[layer_idx + 1]
        elif name.startswith("patch_embedding") or name.startswith("frontend"):
            scale = scales[0]
        else:
            scale = 1.0

        group_lr = base_lr * scale
        target_dict = no_decay_groups if is_no_decay else param_groups
        target_dict.setdefault(group_lr, []).append(param)

    groups: list[dict[str, Any]] = []
    for lr_val, params in param_groups.items():
        groups.append({"params": params, "lr": lr_val, "weight_decay": weight_decay})
    for lr_val, params in no_decay_groups.items():
        groups.append({"params": params, "lr": lr_val, "weight_decay": 0.0})
    return groups


class FTBeatTrainer(BEATTrainer):
    def __init__(
        self,
        model: BaseModel,
        config: FTBeatTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        separator: FRCRNSeparator | None = None,
        target_rms: float = 0.07925,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            config=config,
            device=device,
            logger=logger,
            name=name,
            checkpoint_path=checkpoint_path,
            separator=separator,
            resume_path=resume_path,
            drive_backup=drive_backup,
            on_checkpoint=on_checkpoint,
        )
        self.target_rms = target_rms
        self.accumulation_steps = config.accumulation_steps
        self.amp_mode = config.amp
        self.selection_metric = config.selection_metric

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        layer_decay = getattr(self.config, "layer_decay", 1.0)
        if layer_decay < 1.0:
            params = get_layer_decay_params(
                self.model,
                self.config.lr,
                self.config.weight_decay,
                layer_decay,
            )
            self.optimizer = AdamW(params, betas=tuple(self.config.betas))
        else:
            self.optimizer = AdamW(
                self.trainable_parameters,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=tuple(self.config.betas),
            )
        effective_steps = max(1, steps_per_epoch // self.accumulation_steps)
        total_steps = self.config.epochs * effective_steps
        self.scheduler = cosine_with_warmup(
            self.optimizer, total_steps, self.config.warmup_ratio, self.config.min_lr_ratio
        )
        self._apply_resume_state()

    def prepare(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        waveform = batch["waveform"].to(self.device)
        labels = batch["target"].argmax(dim=1).to(self.device)
        if self.separator is not None:
            with torch.no_grad():
                waveform = self.separator(waveform)
        if self.target_rms > 0:
            waveform = rms_normalize(waveform, self.target_rms)
        return waveform, labels

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform, labels = self.prepare(batch)
        with autocast_context(self.device, self.amp_mode):
            logits = self.model(waveform).logits
            loss = self.criterion(logits, labels)
        self._labels, self._logits = labels, logits.detach().float()
        return StepOutput(loss=loss, size=len(labels))

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        if epoch <= self._completed_epochs:
            return {}
        self._current_epoch = epoch
        self._in_fit = True
        self.model.train()
        self.begin_epoch()
        total_loss, count = 0.0, 0

        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            loader,
            desc=f"{self.name} - epoch {epoch}/{self.config.epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(progress):
            step = self.training_step(batch)
            if not torch.isfinite(step.loss):
                continue
            scaled_loss = step.loss / self.accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                clip_grad_norm_(self.trainable_parameters, self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            self.observe(step)
            total_loss += step.loss.item() * step.size
            count += step.size
            progress.set_postfix(loss=f"{total_loss / max(count, 1):.3f}", lr=f"{self.lr:.1e}")

        return {"train_loss": total_loss / max(count, 1), **self.epoch_summary()}

    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        scores = self._evaluate_with(self.model.predict_proba, loader, desc)
        if self._in_fit and self._current_epoch > 0:
            self._history.append(
                {"epoch": self._current_epoch, **{f"val_{key}": value for key, value in scores.items()}}
            )
            if self.resume_path is not None:
                self.save_resume(self.resume_path)
                if self.drive_backup is not None:
                    self.drive_backup(self.resume_path)
        return scores

    @torch.no_grad()
    def _collect(
        self, predict_proba: Callable[[Tensor], Tensor], loader: DataLoader, desc: str
    ) -> ClassificationMetrics:
        metrics = ClassificationMetrics(num_classes=self.num_classes)
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            waveform, labels = self.prepare(batch)
            with autocast_context(self.device, self.amp_mode):
                probs = predict_proba(waveform).float()
            metrics.update(labels, probs)
        return metrics
