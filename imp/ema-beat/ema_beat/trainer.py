from __future__ import annotations

import copy
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import BaseModel
from src.metrics import ClassificationMetrics
from src.model.frcrn import FRCRNSeparator
from src.train import BEATTrainer, StepOutput, TrainingResult
from src.train.beat_trainer import cosine_with_warmup

from .config import EMAConfig, EMABeatTrainConfig
from .ema import BEATsModelEMA
from .loss import CRDLoss, StudentTeacherEMALoss


class EMABeatTrainer(BEATTrainer):
    """Trainer for Student-Teacher BEATs interaction via Exponential Moving Average (EMA).

    - Student: Active trainable model receiving mix through FRCRN masking (cache-nhat).
    - Teacher: Shadow weights model (EMA), continuously updated via Polyak averaging from Student and providing distillation targets.
    - Supports evaluation on AudioSet (36 classes) and cache-nhat.
    - Integrated Early Stopping, Dynamic LR and ReduceLROnPlateau.
    """

    def __init__(
        self,
        model: BaseModel,
        config: EMABeatTrainConfig,
        ema_config: EMAConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        teacher_model: nn.Module | None = None,
        separator: FRCRNSeparator | None = None,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
        audioset_val_loader: DataLoader | None = None,
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
        self.ema_config = ema_config
        self.selection_metric = config.selection_metric
        self.selection_target = ema_config.selection_target
        self.audioset_val_loader = audioset_val_loader

        # Module EMA (Teacher Shadow Model): khởi tạo từ teacher_model (đã nạp pretrain)
        self.ema = BEATsModelEMA(
            model=model,
            teacher_model=teacher_model,
            decay=ema_config.decay,
            warmup_steps=ema_config.warmup_steps,
            warmup_decay_start=ema_config.warmup_decay_start,
            update_interval=ema_config.update_interval,
            device=device,
            cpu_offload=ema_config.cpu_offload,
        )

        # CRD Module (Contrastive Representation Distillation)
        crd_loss: CRDLoss | None = None
        if getattr(ema_config, "crd_weight", 0.0) > 0.0:
            student_dim = getattr(getattr(model, "config", None), "encoder_embed_dim", 768)
            teacher_dim = student_dim
            crd_loss = CRDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                num_samples=getattr(ema_config, "crd_num_samples", 30240),
                embed_dim=getattr(ema_config, "crd_dim", 128),
                negatives=getattr(ema_config, "crd_negatives", 4096),
                temperature=getattr(ema_config, "crd_temperature", 0.07),
                momentum=getattr(ema_config, "crd_momentum", 0.5),
            )
            crd_loss.to(device)

        # Student-Teacher Loss with Knowledge Distillation and CRD
        self.criterion = StudentTeacherEMALoss(
            kd_weight=ema_config.kd_weight,
            temperature=ema_config.temperature,
            label_smoothing=config.label_smoothing,
            crd_weight=getattr(ema_config, "crd_weight", 0.0),
            crd_loss=crd_loss,
        )
        self.criterion.to(device)

        self._last_raw_scores: dict[str, float] = {}
        self._last_ema_scores: dict[str, float] = {}
        self._last_audioset_scores: dict[str, float] = {}
        self._best_score: float = -float("inf")
        self._best_epoch: int = 0
        self._stale_count: int = 0
        self._plateau_scheduler: ReduceLROnPlateau | None = None

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """All trainable parameters including Student model and CRD projection heads."""
        params = [p for p in self.checkpoint_module.parameters() if p.requires_grad]
        if hasattr(self, "criterion") and self.criterion is not None:
            params.extend([p for p in self.criterion.parameters() if p.requires_grad])
        return params

    def get_parameter_groups(self) -> list[dict[str, Any]]:
        """Construct parameter groups with optional Layer-wise Learning Rate Decay (LLRD)."""
        layer_decay = getattr(self.config, "layer_decay", 1.0)
        base_lr = self.config.lr
        weight_decay = self.config.weight_decay

        if layer_decay >= 1.0 or layer_decay <= 0.0:
            return [{"params": self.trainable_parameters, "lr": base_lr, "weight_decay": weight_decay}]

        num_layers = getattr(getattr(self.model, "config", None), "encoder_layers", 12)
        param_groups_dict: dict[float, list[torch.nn.Parameter]] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head."):
                scale = 1.0
            elif "encoder.layers." in name:
                layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
                scale = layer_decay ** (num_layers - 1 - layer_idx)
            elif "backbone." in name:
                scale = layer_decay ** num_layers
            else:
                scale = 1.0

            param_groups_dict.setdefault(scale, []).append(param)

        # Include trainable parameters from criterion (e.g. CRD projection heads) at base learning rate (scale 1.0)
        if hasattr(self, "criterion") and self.criterion is not None:
            crd_params = [p for p in self.criterion.parameters() if p.requires_grad]
            if crd_params:
                param_groups_dict.setdefault(1.0, []).extend(crd_params)

        groups = []
        for scale, params in sorted(param_groups_dict.items(), key=lambda x: x[0], reverse=True):
            groups.append({
                "params": params,
                "lr": base_lr * scale,
                "initial_lr": base_lr * scale,
                "weight_decay": weight_decay,
            })

        self.logger.info(
            "[%s] Configured Layer-wise LR Decay (factor=%.2f) with %d parameter groups (base_lr=%.2e)",
            self.name, layer_decay, len(groups), base_lr,
        )
        return groups

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        param_groups = self.get_parameter_groups()
        self.optimizer = AdamW(
            param_groups,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=tuple(self.config.betas),
        )
        total_steps = self.config.epochs * steps_per_epoch
        self.scheduler = cosine_with_warmup(
            self.optimizer, total_steps, self.config.warmup_ratio, self.config.min_lr_ratio
        )
        self._apply_resume_state()

        # Setup ReduceLROnPlateau if configured
        if getattr(self.config, "scheduler", "") in ("plateau", "warmup_plateau"):
            self._plateau_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=getattr(self.config, "plateau_factor", 0.5),
                patience=getattr(self.config, "plateau_patience", 2),
                min_lr=getattr(self.config, "min_lr", 1.0e-6),
            )

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        """Run student forward pass, receive teacher EMA distillation, and compute loss."""
        waveform, labels = self.prepare(batch)

        # 1. Teacher EMA inference (no grad) to produce soft targets and representations.
        #    Teacher sees the clean AudioSet noise clip when the batch provides one;
        #    Student sees the FRCRN-masked mixture.
        teacher_waveform = batch.get("teacher_waveform")
        teacher_waveform = teacher_waveform.to(self.device) if teacher_waveform is not None else waveform
        with torch.no_grad():
            teacher_output = self.ema.forward(teacher_waveform)
            teacher_logits = teacher_output.logits if hasattr(teacher_output, "logits") else teacher_output
            teacher_features = getattr(teacher_output, "features", None)

        # 2. Student forward pass
        output = self.model(waveform)
        self._logits = output.logits.detach()
        self._labels = labels.detach()
        student_features = getattr(output, "features", None)

        # 3. Compute Student-Teacher Loss (Task + Logit KD + CRD)
        index = batch.get("index")
        loss, details = self.criterion(
            student_logits=output.logits,
            targets=labels,
            teacher_logits=teacher_logits,
            student_features=student_features,
            teacher_features=teacher_features,
            index=index,
        )

        # 4. Record metrics
        batch_size = waveform.shape[0]
        if labels.ndim == 1:
            correct = (output.logits.argmax(dim=-1) == labels).sum().item()
        else:
            preds = (torch.sigmoid(output.logits) > 0.5).float()
            correct = (preds == labels).all(dim=-1).sum().item()
        self._correct += correct
        self._seen += batch_size

        return StepOutput(loss=loss, size=batch_size)

    def observe(self, step: StepOutput) -> None:
        """Update Teacher EMA weights after optimizer step."""
        super().observe(step)
        if self.ema is not None:
            self.ema.update(self.model)

    def on_no_improvement(self, epoch: int, stale: int) -> None:
        """Reduce Learning Rate when hitting Plateau."""
        plateau_patience = getattr(self.config, "plateau_patience", 2)
        plateau_factor = getattr(self.config, "plateau_factor", 0.5)
        min_lr = getattr(self.config, "min_lr", 1.0e-6)

        if plateau_factor > 0.0 and stale > 0 and (stale % plateau_patience == 0):
            current_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = max(current_lr * plateau_factor, min_lr)
            if new_lr < current_lr:
                for group in self.optimizer.param_groups:
                    group["lr"] = new_lr
                if hasattr(self.scheduler, "base_lrs"):
                    self.scheduler.base_lrs = [max(lr * plateau_factor, min_lr) for lr in self.scheduler.base_lrs]
                self.logger.info(
                    "[%s] Plateau detected (stale %d/%d): Reduced learning rate %.2e -> %.2e",
                    self.name, stale, self.config.patience, current_lr, new_lr
                )

    def epoch_summary(self) -> dict[str, float]:
        summary = super().epoch_summary()
        if self.ema is not None:
            summary["weight_distance"] = self.ema.weight_distance(self.model)
            summary["ema_decay"] = self.ema.get_decay()
        return summary

    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        """Dual evaluation of Student, Teacher EMA on cache-nhat and AudioSet."""
        self.model.eval()

        # 1. Evaluate Student (Raw Model) on cache-nhat
        raw_scores = self._evaluate_with(self.model.predict_proba, loader, desc=f"{desc} (student)")
        self._last_raw_scores = raw_scores

        # 2. Evaluate Teacher EMA on cache-nhat
        if self.ema_config.evaluate_ema and self.ema is not None:
            ema_scores = self._evaluate_with(self.ema.predict_proba, loader, desc=f"{desc} (teacher-ema)")
            self._last_ema_scores = ema_scores
        else:
            ema_scores = raw_scores
            self._last_ema_scores = ema_scores

        # 3. Evaluate Teacher EMA on AudioSet (clean domain) if loader available
        if self.audioset_val_loader is not None and self.ema is not None:
            audioset_scores = self._evaluate_with(
                self.ema.predict_proba, self.audioset_val_loader, desc=f"{desc} (teacher-audioset)"
            )
            self._last_audioset_scores = audioset_scores
        else:
            audioset_scores = {}

        # Record detailed history
        if self._in_fit and self._current_epoch > 0:
            history_entry: dict[str, Any] = {
                "epoch": self._current_epoch,
                **{f"val_student_{k}": v for k, v in raw_scores.items()},
                **{f"val_teacher_{k}": v for k, v in ema_scores.items()},
                **{f"val_raw_{k}": v for k, v in raw_scores.items()},
                **{f"val_ema_{k}": v for k, v in ema_scores.items()},
                **{f"val_audioset_{k}": v for k, v in audioset_scores.items()},
                "weight_distance": self.ema.weight_distance(self.model),
                "ema_decay": self.ema.get_decay(),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self._history.append(history_entry)

            if self.resume_path is not None:
                self.save_resume(self.resume_path)
                if self.drive_backup is not None:
                    self.drive_backup(self.resume_path)

        # Primary score
        if self.selection_target == "ema":
            primary_scores = dict(ema_scores)
            primary_scores["student_macro_f1"] = raw_scores.get("macro_f1", 0.0)
            primary_scores["student_acc"] = raw_scores.get("acc", 0.0)
            return primary_scores
        elif self.selection_target == "model":
            primary_scores = dict(raw_scores)
            primary_scores["teacher_macro_f1"] = ema_scores.get("macro_f1", 0.0)
            primary_scores["teacher_acc"] = ema_scores.get("acc", 0.0)
            return primary_scores
        else:
            return {
                k: 0.5 * (raw_scores.get(k, 0.0) + ema_scores.get(k, 0.0))
                for k in set(raw_scores) | set(ema_scores)
            }

    def snapshot(self) -> dict[str, Tensor]:
        """Create snapshot of best model weights."""
        if self.selection_target == "ema" and self.ema is not None:
            return {k: v.clone().detach() for k, v in self.ema.ema_model.state_dict().items()}
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def restore_best(self) -> None:
        """Restore best checkpoint weights into Student and Teacher."""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            if self.ema is not None:
                self.ema.ema_model.load_state_dict(self._best_state)
            self.logger.info("Restored best checkpoint state (score=%.4f, epoch=%d)", self._best_score, self._best_epoch)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        """Fit model with Student-Teacher distillation, dynamically sizing CRD memory bank if needed."""
        if (
            hasattr(self, "criterion")
            and getattr(self.criterion, "crd", None) is not None
            and hasattr(train_loader, "dataset")
        ):
            try:
                num_train = len(train_loader.dataset)
                if num_train > 0 and num_train != self.criterion.crd.bank_student.shape[0]:
                    self.criterion.crd.reset_bank(num_train)
                    self.criterion.crd.to(self.device)
            except (TypeError, AttributeError):
                pass
        return super().fit(train_loader, val_loader)

    def save_resume(self, path: str | Path | None = None) -> Path:
        """Save complete resume checkpoint with model, shadow EMA, optimizer and scheduler."""
        if self.optimizer is None or self._current_epoch == 0:
            raise ValueError("configure_optimizer and run_epoch must run before save_resume")
        path = Path(path) if path else self.resume_path
        if path is None:
            raise ValueError("resume_path is not set")

        payload = {
            "epoch": self._current_epoch,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "criterion": self.criterion.state_dict() if self.criterion is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_score": self._best_score,
            "best_epoch": self._best_epoch,
            "best_state": self._best_state,
            "stale": self._stale,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def load_resume(self, path: str | Path) -> None:
        """Load resume checkpoint."""
        super().load_resume(path)
        state = torch.load(path, map_location=self.device)
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])
        if self.criterion is not None and state.get("criterion") is not None:
            crit_state = state["criterion"]
            if getattr(self.criterion, "crd", None) is not None and "crd.bank_student" in crit_state:
                saved_size = crit_state["crd.bank_student"].shape[0]
                if saved_size != self.criterion.crd.bank_student.shape[0]:
                    self.criterion.crd.reset_bank(saved_size)
                    self.criterion.crd.to(self.device)
            self.criterion.load_state_dict(crit_state)

    def export_inference_model(self) -> BaseModel:
        """Export standalone BEATClassifier with Teacher EMA weights."""
        from .ema import clone_model
        inference_model = clone_model(self.model)
        if self.ema is not None:
            inference_model.load_state_dict(self.ema.ema_model.state_dict())
        else:
            inference_model.load_state_dict(self.model.state_dict())
        inference_model.eval()
        return inference_model
