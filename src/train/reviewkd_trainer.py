from __future__ import annotations

import copy
import itertools
import logging
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.loss import CrossEntropyLoss, HierarchicalContextLoss
from src.metrics import ClassificationMetrics
from src.model.baseline import BaselineModel
from src.model.reviewkd import ReviewModule, ReviewTeacher

from .base import StepOutput, Trainer
from .beat_trainer import cosine_with_warmup
from .config import ReviewKDTrainConfig

NINE_METRICS = (
    "top1_accuracy",
    "top3_accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "micro_f1",
    "mAP",
    "macro_auc",
)


def snr_scores(probabilities: np.ndarray, labels: np.ndarray, snrs: np.ndarray, num_classes: int) -> dict[str, float]:
    """Điểm tổng và theo từng mức SNR. Với mỗi mức: đủ 9 chỉ số dưới tên `<chỉ số>_snr_<mức>`, cùng `<chỉ số>_snr_mean`.

    Giữ nguyên các khoá cũ: `acc_snr_*`, `balanced_snr_*`, `mAP_snr_*`, `macro_acc`, `mAP_macro`.
    """
    overall = ClassificationMetrics(num_classes=num_classes)
    overall.update(labels, probabilities)
    scores = overall.compute().to_dict()
    scores["acc"] = scores["top1_accuracy"]
    balanced, mean_ap = [], []
    per_metric: dict[str, list[float]] = {name: [] for name in NINE_METRICS}
    for snr in np.unique(snrs):
        mask = snrs == snr
        per_snr = ClassificationMetrics(num_classes=num_classes)
        per_snr.update(labels[mask], probabilities[mask])
        result = per_snr.compute()
        values = result.to_dict()
        scores[f"acc_snr_{int(snr)}"] = result.top1_accuracy
        scores[f"balanced_snr_{int(snr)}"] = result.balanced_accuracy
        scores[f"mAP_snr_{int(snr)}"] = result.mAP
        for name in NINE_METRICS:
            scores[f"{name}_snr_{int(snr)}"] = values[name]
            per_metric[name].append(values[name])
        balanced.append(result.balanced_accuracy)
        mean_ap.append(result.mAP)
    for name, series in per_metric.items():
        if series:
            scores[f"{name}_snr_mean"] = float(np.mean(series))
    scores["macro_acc"] = float(np.mean(balanced)) if balanced else scores["balanced_accuracy"]
    scores["mAP_macro"] = float(np.mean(mean_ap)) if mean_ap else scores["mAP"]
    return scores


class ReviewKDTrainer(Trainer):
    selection_metric: ClassVar[str] = "macro_acc"

    def __init__(
        self,
        model: BaselineModel,
        review: ReviewModule,
        teacher: ReviewTeacher | None,
        config: ReviewKDTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path, on_checkpoint)
        self.resume_path = Path(resume_path) if resume_path else None
        self.drive_backup = drive_backup
        self._completed_epochs = 0
        self._past_records: list[dict[str, float]] = []
        self._epoch_records: list[dict[str, float]] = []
        self._last_train: dict[str, float] = {}
        self._last_scores: dict[str, float] = {}
        self._resume_state: dict[str, Any] | None = None
        self._resumed_best_state: dict[str, Tensor] | None = None
        self._resumed_stale = 0
        self._teacher_logits: Tensor | None = None
        self.review = review.to(device)
        self.teacher = teacher.to(device) if teacher is not None else None
        self.criterion = CrossEntropyLoss(config.label_smoothing)
        self.hcl = HierarchicalContextLoss(config.hcl_levels)
        self._epoch = 0
        self._shapes_logged = False
        self._reset_stats()

    @property
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return itertools.chain(self.model.parameters(), self.review.parameters())

    @property
    def num_classes(self) -> int:
        return self.model.config.num_classes

    def review_weight(self, epoch: int) -> float:
        warmup = self.config.review_warmup_epochs
        scale = 1.0 if warmup == 0 else min(1.0, (epoch - 1) / warmup)
        return self.config.review_weight * scale

    def logit_weight(self, epoch: int) -> float:
        warmup = self.config.review_warmup_epochs
        scale = 1.0 if warmup == 0 else min(1.0, (epoch - 1) / warmup)
        return self.config.logit_weight * scale

    def _reset_stats(self) -> None:
        self._ce_sum = 0.0
        self._review_sum = 0.0
        self._logit_sum = 0.0
        self._level_sum: Tensor | None = None
        self._seen = 0
        self._review_seen = 0
        self._teacher_seconds = 0.0
        self._teacher_steps = 0
        self._correct = 0

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
        if self._resume_state is not None:
            self.optimizer.load_state_dict(self._resume_state["optimizer"])
            if self._resume_state.get("scheduler") is not None:
                self.scheduler.load_state_dict(self._resume_state["scheduler"])
            self._resume_state = None
            self.logger.info(
                "%s resume: optimizer/scheduler khôi phục từ epoch %d (lr %.1e)",
                self.name,
                self._completed_epochs,
                self.lr,
            )

    def begin_epoch(self) -> None:
        self.review.train()
        if self.teacher is not None:
            self.teacher.eval()
        self._reset_stats()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self._epoch = epoch
        self._last_train = super().run_epoch(loader, epoch)
        return self._last_train

    def skip_epoch(self, epoch: int) -> bool:
        return epoch <= self._completed_epochs

    def initial_stale(self) -> int:
        return self._resumed_stale

    def snapshot(self) -> dict[str, Tensor]:
        if self._resumed_best_state is not None:
            state, self._resumed_best_state = copy.deepcopy(self._resumed_best_state), None
            return state
        return super().snapshot()

    def save_resume(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.resume_path
        if path is None or self.optimizer is None or self._epoch == 0:
            raise ValueError("optimizer, a finished epoch and a resume path are required for save_resume")
        payload = {
            "epoch": self._epoch,
            "model": self.model.state_dict(),
            "review": self.review.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_score": self._best_score,
            "best_epoch": self._best_epoch,
            "best_state": copy.deepcopy(self._best_state) if self._best_state is not None else None,
            "epoch_records": list(self._epoch_records),
            "stale": self._stale,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        torch.save(payload, partial)
        os.replace(partial, path)
        return path

    def load_resume(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(data["model"])
        self.review.load_state_dict(data["review"])
        self._completed_epochs = int(data["epoch"])
        self._epoch = self._completed_epochs
        self._best_score = float(data.get("best_score", -float("inf")))
        self._best_epoch = int(data.get("best_epoch", 0))
        self._resumed_best_state = data.get("best_state")
        self._resumed_stale = int(data.get("stale", 0))
        self._past_records = list(data.get("epoch_records", []))
        self._epoch_records = list(self._past_records)
        self._resume_state = data
        self.logger.info(
            "%s resume: từ epoch %d (best %.4f @ %d, stale %d)",
            self.name,
            self._completed_epochs,
            self._best_score,
            self._best_epoch,
            self._resumed_stale,
        )

    def end_epoch(self, epoch: int) -> None:
        self._epoch_records.append(
            {"epoch": epoch, **self._last_train, **{f"val_{k}": v for k, v in self._last_scores.items()}}
        )
        if self.resume_path is None:
            return
        self.save_resume(self.resume_path)
        if self.drive_backup is not None:
            self.drive_backup(self.resume_path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        result = super().fit(train_loader, val_loader)
        result.history = self._past_records + result.history
        return result

    def _call_teacher(self, waveform: Tensor) -> Tensor:
        """Gọi teacher lấy lưới đặc trưng. Teacher có logit thì đặt `self._teacher_logits` ở đây cho KD logit."""
        return self.teacher(waveform)

    def _teacher_features(self, waveform: Tensor) -> Tensor:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        features = self._call_teacher(waveform)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._teacher_seconds += time.perf_counter() - started
        self._teacher_steps += 1
        return features

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform = batch["waveform"].to(self.device)
        labels = batch["target"].argmax(dim=1).to(self.device)
        weight = self.review_weight(self._epoch)
        logit_weight = self.logit_weight(self._epoch)
        use_review = self.teacher is not None and weight > 0
        use_logit = self.teacher is not None and logit_weight > 0
        output = self.model(waveform, return_features=use_review)
        ce = self.criterion(output.logits, labels)
        loss = ce
        teacher_features = self._teacher_features(waveform) if (use_review or use_logit) else None
        if use_review:
            student_features = self.review(output.features)
            levels = self.hcl.per_level(student_features, teacher_features)
            review = levels.sum()
            loss = ce + weight * review
            self._review_sum += review.item() * len(labels)
            self._review_seen += len(labels)
            detached = levels.detach() * len(labels)
            self._level_sum = detached if self._level_sum is None else self._level_sum + detached
            if not self._shapes_logged:
                self._shapes_logged = True
                self.logger.info(
                    "%s | stage features %s | review %s | teacher %s | ce %.4f | review %.4f | weight %.3f",
                    self.name,
                    [tuple(f.shape) for f in output.features],
                    tuple(student_features.shape),
                    tuple(teacher_features.shape),
                    ce.item(),
                    review.item(),
                    weight,
                )
        if use_logit:
            if self._teacher_logits is None:
                raise ValueError("logit_weight > 0 cần teacher trả logit (NoiseTeacher.forward(..., with_logits=True))")
            temperature = self.config.logit_temperature
            logit_kd = (
                F.kl_div(
                    F.log_softmax(output.logits / temperature, dim=-1),
                    F.softmax(self._teacher_logits.detach().float() / temperature, dim=-1),
                    reduction="batchmean",
                )
                * temperature**2
            )
            loss = loss + logit_weight * logit_kd
            self._logit_sum += logit_kd.item() * len(labels)
        self._ce_sum += ce.item() * len(labels)
        self._seen += len(labels)
        self._correct += (output.logits.argmax(dim=-1) == labels).sum().item()
        return StepOutput(loss=loss, size=len(labels))

    def epoch_summary(self) -> dict[str, float]:
        seen = max(self._seen, 1)
        summary = {
            "acc": self._correct / seen,
            "ce": self._ce_sum / seen,
            "review": self._review_sum / max(self._review_seen, 1),
            "review_weight": self.review_weight(self._epoch),
        }
        if self.config.logit_weight > 0:
            summary["logit_kd"] = self._logit_sum / seen
            summary["logit_weight"] = self.logit_weight(self._epoch)
        if self._level_sum is not None:
            for index, value in enumerate((self._level_sum / max(self._review_seen, 1)).tolist()):
                summary[f"review_l{index + 1}"] = value
        if self._teacher_steps:
            summary["teacher_s_per_step"] = self._teacher_seconds / self._teacher_steps
        if self.device.type == "cuda":
            summary["peak_vram_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        return summary

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        probabilities, labels, snrs = [], [], []
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            waveform = batch["waveform"].to(self.device)
            probabilities.append(self.model.predict_proba(waveform).float().cpu())
            labels.append(batch["target"].argmax(dim=1).cpu())
            snrs.append(batch["snr_db"].cpu())
        self._last_scores = snr_scores(
            torch.cat(probabilities).numpy(),
            torch.cat(labels).numpy(),
            torch.cat(snrs).numpy(),
            self.num_classes,
        )
        return self._last_scores
