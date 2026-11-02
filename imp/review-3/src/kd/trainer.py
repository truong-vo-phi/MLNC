"""
Knowledge Distillation Trainer for Task 3:
Supports teacher-student training with 5 KD methods,
epoch logging, 6 metrics tracking, and detailed SNR breakdowns.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.kd.losses import CompositeKDLoss
from src.metrics import compute_metrics, compute_snr_breakdown
from src.utils import count_parameters, save_checkpoint


class KDTrainer:
    """
    Knowledge Distillation Trainer for BEATs -> CED-Tiny transfer.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        method: str = "logit",
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        checkpoint_dir: str | Path = "checkpoints/kd",
        run_name: str = "kd_run",
    ) -> None:
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.method = method
        self.run_name = run_name
        self.logger = logger or logging.getLogger("kd_trainer")
        self.checkpoint_dir = Path(checkpoint_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = self.config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 25))
        self.lr = float(train_cfg.get("lr", 3.0e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.05))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.patience = int(train_cfg.get("patience", 7))
        self.metric_for_best = self.config.get("logging", {}).get("metric_for_best", "macro_f1")

        # Loss function
        self.loss_fn = CompositeKDLoss(
            method=method,
            student_dim=192,
            teacher_dim=768,
            num_classes=self.config.get("dataset", {}).get("num_classes", 36),
            temperature=float(train_cfg.get("kd_temperature", 2.0)),
            alpha=float(train_cfg.get("lambda_kd", 0.7)),
            beta=float(train_cfg.get("lambda_crd", 0.5)),
            gamma=float(train_cfg.get("lambda_mgd", 0.5)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
        ).to(self.device)

        # Optimizer: optimizes student and any auxiliary adaptation layers in loss_fn
        trainable_params = list(self.student.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        )

        # LR Scheduler
        warmup_epochs = train_cfg.get("warmup_epochs", 2)
        total_steps = max(1, self.epochs * len(train_loader))
        warmup_steps = max(1, warmup_epochs * len(train_loader))
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.best_metric = -1.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

    def _log_header(self) -> None:
        total_s, train_s = count_parameters(self.student)
        total_t, _ = count_parameters(self.teacher)
        self.logger.info("=" * 86)
        self.logger.info("KNOWLEDGE DISTILLATION: TEACHER (BEATs) -> STUDENT (%s)", self.run_name.upper())
        self.logger.info("KD Method: %s | Device: %s", self.method.upper(), self.device)
        self.logger.info("Teacher Parameters: %.2fM (Frozen)", total_t / 1e6)
        self.logger.info("Student Parameters: Total=%.2fM, Trainable=%.2fM", total_s / 1e6, train_s / 1e6)
        self.logger.info("Epochs: %d | Batch Size: %d | Initial LR: %.1e",
                         self.epochs, self.train_loader.batch_size, self.lr)
        self.logger.info("=" * 86)

    def train_epoch(self, epoch: int) -> tuple[float, float, dict[str, float]]:
        self.student.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        comp_loss_totals: dict[str, float] = {}

        for batch in self.train_loader:
            wave = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            bsz = wave.size(0)

            # 1. Forward teacher (frozen)
            with torch.no_grad():
                if hasattr(self.teacher, "forward_features"):
                    t_feat = self.teacher.forward_features(
                        self.teacher.frontend(wave).features if hasattr(self.teacher, "frontend") else wave
                    )
                    t_logits = self.teacher.forward_head(t_feat)
                else:
                    t_out = self.teacher(wave)
                    t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
                    t_feat = t_logits

            # 2. Forward student
            if hasattr(self.student, "forward_features"):
                s_feat = self.student.forward_features(
                    self.student.frontend(wave).features if hasattr(self.student, "frontend") else wave
                )
                s_logits = self.student.forward_head(s_feat)
            else:
                s_out = self.student(wave)
                s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
                s_feat = s_logits

            # 3. Compute composite KD loss
            loss, loss_dict = self.loss_fn(
                student_logits=s_logits,
                teacher_logits=t_logits,
                student_feat=s_feat,
                teacher_feat=t_feat,
                targets=target,
            )

            # 4. Backward & step
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.item()) * bsz
            for k, v in loss_dict.items():
                comp_loss_totals[k] = comp_loss_totals.get(k, 0.0) + v * bsz

            preds = s_logits.argmax(dim=-1)
            t_idx = target.argmax(dim=-1) if target.ndim == 2 else target
            total_correct += int((preds == t_idx).sum().item())
            total_samples += bsz

        avg_loss = total_loss / max(1, total_samples)
        avg_acc = total_correct / max(1, total_samples)
        avg_comps = {k: v / max(1, total_samples) for k, v in comp_loss_totals.items()}
        return avg_loss, avg_acc, avg_comps

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
        self.student.eval()
        total_loss = 0.0
        total_samples = 0
        all_logits, all_targets, all_snrs = [], [], []

        for batch in loader:
            wave = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            snr_db = batch["snr_db"].numpy()
            bsz = wave.size(0)

            # Teacher forward
            if hasattr(self.teacher, "forward_features"):
                t_feat = self.teacher.forward_features(
                    self.teacher.frontend(wave).features if hasattr(self.teacher, "frontend") else wave
                )
                t_logits = self.teacher.forward_head(t_feat)
            else:
                t_out = self.teacher(wave)
                t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
                t_feat = t_logits

            # Student forward
            if hasattr(self.student, "forward_features"):
                s_feat = self.student.forward_features(
                    self.student.frontend(wave).features if hasattr(self.student, "frontend") else wave
                )
                s_logits = self.student.forward_head(s_feat)
            else:
                s_out = self.student(wave)
                s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
                s_feat = s_logits

            loss, _ = self.loss_fn(s_logits, t_logits, s_feat, t_feat, target)
            total_loss += float(loss.item()) * bsz
            total_samples += bsz

            all_logits.append(s_logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_snrs.append(snr_db)

        val_loss = total_loss / max(1, total_samples)
        logits_arr = np.concatenate(all_logits, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        snrs_arr = np.concatenate(all_snrs, axis=0)

        num_classes = self.config.get("dataset", {}).get("num_classes", 36)
        metrics = compute_metrics(logits_arr, targets_arr, num_classes=num_classes)
        snr_breakdown = compute_snr_breakdown(logits_arr, targets_arr, snrs_arr, num_classes=num_classes)
        return val_loss, metrics, snr_breakdown

    def fit(self) -> dict[str, Any]:
        self._log_header()
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            train_loss, train_acc, comp_losses = self.train_epoch(epoch)
            val_loss, val_metrics, snr_breakdown = self.evaluate(self.val_loader)
            ep_time = time.time() - ep_start

            # Format log line
            comp_str = " | ".join(f"{k.upper()}: {v:.4f}" for k, v in comp_losses.items() if k != "total")
            self.logger.info(
                "[Epoch %02d/%02d] (%.1fs) Train Loss: %.4f | Train Acc: %5.2f%% | Val Acc: %5.2f%% | "
                "Prec: %.4f | Rec: %.4f | Macro-F1: %.4f | Micro-F1: %.4f | mAP: %.4f [%s]",
                epoch, self.epochs, ep_time, train_loss, train_acc * 100, val_metrics["acc"] * 100,
                val_metrics["precision"], val_metrics["recall"], val_metrics["macro_f1"],
                val_metrics["micro_f1"], val_metrics["mAP"], comp_str
            )

            # SNR breakdown
            self.logger.info("  -> SNR Breakdown (Epoch %02d):", epoch)
            for snr_tag, m in snr_breakdown.items():
                tag = snr_tag if "dB" in snr_tag else f"{snr_tag}dB"
                self.logger.info(
                    "     SNR %6s: Acc=%5.2f%% | Prec=%.4f | Rec=%.4f | Macro-F1=%.4f | Micro-F1=%.4f | mAP=%.4f (n=%d)",
                    tag, m["acc"] * 100, m["precision"], m["recall"], m["macro_f1"], m["micro_f1"], m["mAP"], m.get("count", 0)
                )

            current_metric = val_metrics.get(self.metric_for_best, val_metrics["macro_f1"])
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.student.state_dict(),
                        "best_metric": self.best_metric,
                        "val_metrics": val_metrics,
                        "snr_breakdown": snr_breakdown,
                        "method": self.method,
                        "run_name": self.run_name,
                    },
                    best_path,
                )
                self.logger.info("  [*] Saved best model checkpoint to %s (Metric: %.4f)", best_path, self.best_metric)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info("Early stopping triggered after %d epochs without improvement.", self.patience)
                    break

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "snr_breakdown": snr_breakdown,
            })

        total_time = time.time() - start_time
        self.logger.info("=" * 86)
        self.logger.info("FINISHED KD TRAINING: %s in %.1f minutes", self.run_name.upper(), total_time / 60.0)
        self.logger.info("Best Metric (%s): %.4f at Epoch %d", self.metric_for_best, self.best_metric, self.best_epoch)
        self.logger.info("=" * 86)

        return {
            "method": self.method,
            "run_name": self.run_name,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "val_metrics": self.history[self.best_epoch - 1]["val_metrics"] if self.best_epoch > 0 else {},
            "snr_breakdown": self.history[self.best_epoch - 1]["snr_breakdown"] if self.best_epoch > 0 else {},
            "checkpoint": str(self.checkpoint_dir / "best_model.pt"),
            "training_time_s": round(total_time, 2),
        }

