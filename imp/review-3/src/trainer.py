"""
Trainer Engine for Review-3 Models:
Tracks training progress, computes acc/precision/recall/macro_f1/micro_f1/mAP,
logs detailed epoch updates and per-SNR breakdowns to .log files.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metrics import compute_metrics, compute_snr_breakdown
from src.utils import count_parameters, save_checkpoint


class Trainer:
    """
    Standard Trainer supporting any Review-3 Audio Classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        model_name: str = "model",
    ) -> None:
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger or logging.getLogger("trainer")
        self.checkpoint_dir = Path(checkpoint_dir) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        train_cfg = self.config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.lr = float(train_cfg.get("lr", 3.0e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.05))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
        self.patience = int(train_cfg.get("patience", 7))
        self.metric_for_best = self.config.get("logging", {}).get("metric_for_best", "acc")

        # Criterion with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        )

        # Scheduler: Warmup + Cosine Annealing
        warmup_epochs = train_cfg.get("warmup_epochs", 2)
        total_steps = self.epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)
        min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Tracking state
        self.best_metric = -1.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

    def _log_header(self) -> None:
        total_p, train_p = count_parameters(self.model)
        self.logger.info("=" * 78)
        self.logger.info("STARTING TRAINING PIPELINE: %s", self.model_name.upper())
        self.logger.info("Device: %s | Model Parameters: Total=%d (%.2fM), Trainable=%d (%.2fM)",
                         self.device, total_p, total_p / 1e6, train_p, train_p / 1e6)
        self.logger.info("Configuration: %s", json.dumps(self.config, indent=2, default=str))
        self.logger.info("=" * 78)

    def _log_file_only(self, msg: str, *args: Any) -> None:
        """Logs message strictly to FileHandlers so stdout/live UI stays clean."""
        has_file_handler = False
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler):
                record = self.logger.makeRecord(self.logger.name, logging.INFO, "(file)", 0, msg, args, None)
                h.handle(record)
                has_file_handler = True
        if not has_file_handler:
            self.logger.debug(msg, *args)

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(waveform)
            loss = self.criterion(logits, target)

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.item()) * waveform.size(0)
            preds = logits.argmax(dim=-1)
            target_idx = target.argmax(dim=-1) if target.dim() > 1 else target
            correct += int((preds == target_idx).sum().item())
            total_samples += waveform.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_logits = []
        all_targets = []
        all_snrs = []

        for batch in loader:
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            snr_db = batch["snr_db"].cpu().numpy()

            logits = self.model(waveform)
            loss = self.criterion(logits, target)

            total_loss += float(loss.item()) * waveform.size(0)
            total_samples += waveform.size(0)

            all_logits.append(logits.cpu().numpy())
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
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_metrics, snr_breakdown = self.evaluate(self.val_loader)
            epoch_time = time.time() - epoch_start

            # Format primary log line
            msg = (
                f"[Epoch {epoch:02d}/{self.epochs:02d}] ({epoch_time:.1f}s) "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['acc']*100:.2f}% | "
                f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
                f"Macro-F1: {val_metrics['macro_f1']:.4f} | Micro-F1: {val_metrics['micro_f1']:.4f} | "
                f"mAP: {val_metrics['mAP']:.4f}"
            )
            self.logger.info(msg)

            # Format per-SNR breakdown (recorded to log file only as requested)
            self._log_file_only("  -> SNR Breakdown (Epoch %02d):", epoch)
            for snr_tag, m in snr_breakdown.items():
                self._log_file_only(
                    "     SNR %6s: Acc=%5.2f%% | Prec=%.4f | Rec=%.4f | Macro-F1=%.4f | Micro-F1=%.4f | mAP=%.4f",
                    snr_tag,
                    m["acc"] * 100,
                    m["precision"],
                    m["recall"],
                    m["macro_f1"],
                    m["micro_f1"],
                    m["mAP"],
                )

            # Record history
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "snr_breakdown": snr_breakdown,
                "epoch_time_s": epoch_time,
            }
            self.history.append(record)

            # Checkpoint best model
            if self.metric_for_best not in val_metrics:
                raise ValueError(f"Unknown checkpoint metric: {self.metric_for_best}")
            current_metric = val_metrics[self.metric_for_best]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                        "val_metrics": val_metrics,
                        "snr_breakdown": snr_breakdown,
                        "config": self.config,
                    },
                    best_path,
                )
                self.logger.info("  [*] New best model saved to %s (Metric: %.4f)", best_path, self.best_metric)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info("Early stopping triggered after %d epochs without improvement.", self.patience)
                    break

        total_time = time.time() - start_time
        self.logger.info("=" * 78)
        self.logger.info("TRAINING FINISHED: %s in %.1f minutes", self.model_name, total_time / 60.0)
        self.logger.info("Best Epoch: %d with %s = %.4f", self.best_epoch, self.metric_for_best, self.best_metric)

        # Final evaluation on test loader if provided
        test_summary = {}
        if self.test_loader is not None:
            self.logger.info("Running final evaluation on Test Split...")
            best_ckpt = self.checkpoint_dir / "best_model.pt"
            if best_ckpt.is_file():
                ckpt = torch.load(best_ckpt, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
            test_loss, test_metrics, test_snr = self.evaluate(self.test_loader)
            self.logger.info("Test Loss: %.4f | Acc: %.2f%% | Macro-F1: %.4f | mAP: %.4f",
                             test_loss, test_metrics['acc'] * 100, test_metrics['macro_f1'], test_metrics['mAP'])
            test_summary = {"test_loss": test_loss, "test_metrics": test_metrics, "test_snr": test_snr}

        # Save history json
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({"history": self.history, "test": test_summary, "best_metric": self.best_metric}, f, indent=2)

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "history": self.history,
            "test": test_summary,
        }
