from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import BaseModel
from src.metrics import ClassificationMetrics
from src.model.frcrn import FRCRNSeparator
from src.train import StepOutput, Trainer
from src.train.beat_trainer import cosine_with_warmup
from src.train.reviewkd_trainer import snr_scores
from src.utils.checkpoint import save_checkpoint

from .config import CEDTrainConfig
from .loss import SoftTargetCrossEntropy
from .model import rms_normalize
from .runtime import autocast_context


class CEDTinyTrainer(Trainer):
    selection_metric = "macro_acc"

    def __init__(
        self,
        model: BaseModel,
        config: CEDTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        separator: FRCRNSeparator | None = None,
        target_rms: float = 0.07925,
        resume_path: str | Path | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            config=config,
            device=device,
            logger=logger,
            name=name,
            checkpoint_path=checkpoint_path,
            on_checkpoint=on_checkpoint,
        )
        self.separator = separator
        self.target_rms = target_rms
        self.resume_path = Path(resume_path) if resume_path is not None else None
        self.accumulation_steps = max(config.accumulation_steps, 1)
        self.amp_mode = config.amp
        self.selection_metric = config.selection_metric
        self.criterion = SoftTargetCrossEntropy(label_smoothing=config.label_smoothing)
        self.mixup = config.mixup
        self.mixup_alpha = config.mixup_alpha
        self.history: list[dict[str, float]] = []
        self._completed_epochs = 0
        self._in_fit = False
        self._labels: Tensor | None = None
        self._logits: Tensor | None = None

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        fast_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("rel_bias", "grep_", "cls_to_", "patches_to_")):
                fast_params.append(param)
            else:
                base_params.append(param)

        if fast_params:
            param_groups = [
                {"params": base_params, "lr": self.config.lr, "weight_decay": self.config.weight_decay},
                {"params": fast_params, "lr": self.config.lr * 1.5, "weight_decay": 0.0},
            ]
        else:
            param_groups = [{"params": base_params, "lr": self.config.lr, "weight_decay": self.config.weight_decay}]

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.lr,
            betas=tuple(self.config.betas),
        )
        effective_steps = max(1, steps_per_epoch // self.accumulation_steps)
        total_steps = self.config.epochs * effective_steps
        self.scheduler = cosine_with_warmup(
            self.optimizer, total_steps, self.config.warmup_ratio, self.config.min_lr_ratio
        )
        if self.resume_path and self.resume_path.is_file():
            self.load_resume(self.resume_path)

    def prepare(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        waveform = batch["waveform"].to(self.device)
        target = batch["target"].to(self.device).float()
        if self.separator is not None:
            with torch.no_grad():
                waveform = self.separator(waveform)
        if self.target_rms > 0:
            waveform = rms_normalize(waveform, self.target_rms)
        if self.model.training and self.mixup > 0 and torch.rand(()).item() < self.mixup:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample((waveform.shape[0], 1))
            lam = lam.to(self.device)
            lam = torch.maximum(lam, 1.0 - lam)
            order = torch.randperm(waveform.shape[0], device=self.device)
            waveform = lam * waveform + (1.0 - lam) * waveform[order]
            target = lam * target + (1.0 - lam) * target[order]
        return waveform, target

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform, target = self.prepare(batch)
        with autocast_context(self.device, self.amp_mode):
            logits = self.model(waveform).logits
            loss = self.criterion(logits, target)
        self._labels = target.argmax(dim=-1)
        self._logits = logits.detach().float()
        return StepOutput(loss=loss, size=len(target))

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        if epoch <= self._completed_epochs:
            return {}
        self.model.train()
        self.begin_epoch()
        total_loss, count = 0.0, 0
        correct = 0

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
            scaled = step.loss / self.accumulation_steps
            scaled.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                if self.config.grad_clip > 0:
                    clip_grad_norm_(self.trainable_parameters, self.config.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            bs = step.size
            total_loss += step.loss.item() * bs
            count += bs
            if self._logits is not None and self._labels is not None:
                correct += (self._logits.argmax(dim=-1) == self._labels).sum().item()

            avg_loss = total_loss / max(count, 1)
            acc = correct / max(count, 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}", lr=f"{self.lr:.2e}")

        self._completed_epochs = epoch
        train_metrics = {
            "train_loss": total_loss / max(count, 1),
            "train_acc": correct / max(count, 1),
            "lr": self.lr,
        }
        if self.resume_path is not None and epoch % self.config.resume_every == 0:
            self.save_resume(self.resume_path, epoch)
        return train_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        if self.separator is not None:
            self.separator.eval()

        all_probs: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_snrs: list[Tensor] = []

        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            waveform = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            labels = target.argmax(dim=1).cpu() if target.ndim > 1 else target.cpu()
            snr = batch["snr_db"].cpu()

            if self.separator is not None:
                waveform = self.separator(waveform)
            if self.target_rms > 0:
                waveform = rms_normalize(waveform, self.target_rms)

            with autocast_context(self.device, self.amp_mode):
                probs = self.model.predict_proba(waveform).float().cpu()

            all_probs.append(probs)
            all_labels.append(labels)
            all_snrs.append(snr)

        cat_probs = torch.cat(all_probs, dim=0).numpy()
        cat_labels = torch.cat(all_labels, dim=0).numpy()
        cat_snrs = torch.cat(all_snrs, dim=0).numpy()

        num_classes = self.model.config.num_classes
        cm = ClassificationMetrics(num_classes=num_classes)
        cm.update(torch.from_numpy(cat_labels), torch.from_numpy(cat_probs))
        metrics = cm.compute().to_dict()

        scores = snr_scores(cat_probs, cat_labels, cat_snrs, num_classes)
        metrics.update(scores)
        return metrics

    def restore_best(self) -> None:
        if self.checkpoint_path.is_file():
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("model", payload)
            self.checkpoint_module.load_state_dict(state_dict)
            self.logger.info("Restored best model checkpoint from %s", self.checkpoint_path)

    def save_resume(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.checkpoint_module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "completed_epochs": epoch,
                "best_score": self._best_score,
                "best_epoch": self._best_epoch,
                "stale": self._stale,
                "history": self.history,
            },
            path,
        )

    def load_resume(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.checkpoint_module.load_state_dict(payload["model"])
        if self.optimizer is not None and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and "scheduler" in payload and payload["scheduler"] is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self._completed_epochs = payload.get("completed_epochs", 0)
        self._best_score = payload.get("best_score", -float("inf"))
        self._best_epoch = payload.get("best_epoch", 0)
        self._stale = payload.get("stale", 0)
        self.history = payload.get("history", [])
        self.logger.info(
            "Loaded resume state from %s (completed_epochs=%d, best_score=%.4f)",
            path,
            self._completed_epochs,
            self._best_score,
        )

    def _save_checkpoint(self, score: float, epoch: int) -> None:
        save_checkpoint(
            self.checkpoint_path,
            self.checkpoint_module,
            epoch=epoch,
            score=score,
            selection_metric=self.selection_metric,
        )
        if self.on_checkpoint is not None:
            self.on_checkpoint(self.checkpoint_path, epoch, score)

    @staticmethod
    def _format(stats: dict[str, float]) -> str:
        return " ".join(f"{k} {v:.2e}" if k == "lr" else f"{k} {v:.3f}" for k, v in stats.items())
