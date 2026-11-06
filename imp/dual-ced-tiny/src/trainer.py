"""
Trainer Engine for Dual-CED-Tiny:
Manages 15-epoch training loop, gradient accumulation, decoupled backbone/head learning rates,
cosine learning rate schedule, bfloat16 AMP, evaluation across 6 SNR levels,
and validation accuracy checkpoint selection.
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

from src.metrics import compute_metrics, compute_snr_breakdown
from src.frontend import SpecAugment
from src.utils import count_parameters, save_checkpoint


class Trainer:
    """
    Standard Training Engine for Dual-CED-Tiny and its Expert Sub-models.
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
        model_name: str = "dual_ced_tiny",
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
        self.epochs = train_cfg.get("epochs", 15)
        self.accumulation = train_cfg.get("gradient_accumulation", 2)
        self.backbone_lr = float(train_cfg.get("backbone_lr", 3.0e-5))
        self.head_lr = float(train_cfg.get("head_lr", 3.0e-4))
        self.router_lr = float(train_cfg.get("router_lr", 2.0e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.grad_clip = float(train_cfg.get("clip_grad_norm", 1.0))
        self.label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
        self.use_amp = (train_cfg.get("amp", "bfloat16") == "bfloat16") and (self.device.type == "cuda")
        self.early_stopping = bool(train_cfg.get("early_stopping", True))
        raw_patience = train_cfg.get("patience", 5)
        self.patience: int | None = (
            int(raw_patience) if (self.early_stopping and raw_patience is not None and int(raw_patience) > 0) else None
        )

        # Criterion: CrossEntropy on raw logits
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        front_cfg = self.config.get("frontend", {}) or {}
        self.spec_augment = SpecAugment(
            freq_masks=int(front_cfg.get("specaug_freq_masks", 2)),
            freq_width=int(front_cfg.get("specaug_freq_width", 8)),
            time_masks=int(front_cfg.get("specaug_time_masks", 2)),
            time_width=int(front_cfg.get("specaug_time_width", 24)),
        ).to(self.device)
        self.use_specaugment = bool(train_cfg.get("specaugment", True))

        # Keep the learned router independently tunable, like imp/moe-v2.
        backbone_params, head_params, router_params = [], [], []
        head_identifiers = ("head.", "fc.", "classifier.")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("router.", "acoustic_extractor.")):
                router_params.append(param)
            elif any(h in name for h in head_identifiers):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})
        if router_params:
            param_groups.append({"params": router_params, "lr": self.router_lr})
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": self.backbone_lr}]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler: Cosine with 2 Epochs Warmup
        steps_per_epoch = math.ceil(len(self.train_loader) / self.accumulation)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Checkpoint selection: Max validation acc, earliest epoch on ties
        self.best_metric = -1.0
        self.best_epoch = -1
        self.history: list[dict[str, Any]] = []

    def _prepare_wave(self, wave: torch.Tensor) -> torch.Tensor:
        """Hook for input preprocessing (identity here; KDTrainer applies RMS normalization)."""
        return wave

    def _log_header(self) -> None:
        total_p, train_p = count_parameters(self.model)
        self.logger.info("=" * 80)
        self.logger.info("STARTING TRAINING PIPELINE: %s", self.model_name.upper())
        self.logger.info("Device: %s | Model Parameters: Total=%d (%.2fM), Trainable=%d (%.2fM)",
                         self.device, total_p, total_p / 1e6, train_p, train_p / 1e6)
        self.logger.info("Configuration: %s", json.dumps(self.config, indent=2, default=str))
        self.logger.info("=" * 80)

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        self.spec_augment.train(self.use_specaugment)
        total_loss = 0.0
        correct = 0
        total_samples = 0

        self.optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(self.train_loader):
            wave = self._prepare_wave(batch["waveform"].to(self.device, non_blocking=True))
            target = batch["target"].to(self.device, non_blocking=True)
            snr = batch.get("snr_db")
            if snr is not None:
                snr = snr.to(self.device, non_blocking=True)

            target_idx = target.argmax(dim=-1) if target.ndim == 2 else target

            # MoE-v2 baseline: fused CE + expert specialization + train-only SNR router target.
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                if hasattr(self.model, "forward_from_features") and snr is not None:
                    spec = self.model.frontend(wave)
                    if self.use_specaugment:
                        spec = self.spec_augment(spec)
                    details = self.model.forward_from_features(wave, spec, return_details=True)
                    logits = details["logits"]
                    ce = self.criterion(logits, target_idx)
                    low_ce = nn.functional.cross_entropy(
                        details["logits_low"], target_idx, label_smoothing=self.label_smoothing, reduction="none"
                    )
                    high_ce = nn.functional.cross_entropy(
                        details["logits_high"], target_idx, label_smoothing=self.label_smoothing, reduction="none"
                    )
                    w_high = torch.sigmoid(
                        (snr.float() - self.model.router.transition_center_db)
                        / self.model.router.transition_tau
                    )
                    expert_ce = ((1.0 - w_high) * low_ce + w_high * high_ce).mean()
                    route_loss = self.model.router.compute_router_loss(details["alpha"], snr)
                    router_cfg = self.config.get("router", {}) or {}
                    loss = ce + float(router_cfg.get("expert_ce_weight", 0.5)) * expert_ce \
                        + float(router_cfg.get("lambda_route_loss", 0.2)) * route_loss
                else:
                    logits = self.model(wave, snr_db=snr)
                    loss = self.criterion(logits, target_idx)

            loss = loss / self.accumulation
            loss.backward()

            batch_sz = wave.size(0)
            total_loss += loss.item() * self.accumulation * batch_sz
            preds = logits.argmax(dim=-1)
            correct += (preds == target_idx).sum().item()
            total_samples += batch_sz

            if (idx + 1) % self.accumulation == 0 or (idx + 1) == len(self.train_loader):
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, total_samples)
        avg_acc = correct / max(1, total_samples)
        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float], dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        all_snrs = []
        total_samples = 0

        for batch in loader:
            wave = self._prepare_wave(batch["waveform"].to(self.device, non_blocking=True))
            target = batch["target"].to(self.device, non_blocking=True)
            snr = batch.get("snr_db")
            if snr is not None:
                snr = snr.to(self.device, non_blocking=True)

            target_idx = target.argmax(dim=-1) if target.ndim == 2 else target

            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(wave, snr_db=snr)
                loss = self.criterion(logits, target_idx)

            batch_sz = wave.size(0)
            total_loss += loss.item() * batch_sz
            total_samples += batch_sz

            all_logits.append(logits.float().cpu())
            all_targets.append(target_idx.cpu())
            if snr is not None:
                all_snrs.append(snr.cpu())

        val_loss = total_loss / max(1, total_samples)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_snrs = torch.cat(all_snrs, dim=0) if all_snrs else None

        metrics = compute_metrics(cat_logits, cat_targets, num_classes=self.model.num_classes)
        breakdown = {}
        if cat_snrs is not None:
            breakdown = compute_snr_breakdown(
                cat_logits,
                cat_targets,
                cat_snrs,
                num_classes=self.model.num_classes,
            )

        return val_loss, metrics, breakdown

    def fit(self) -> dict[str, Any]:
        self._log_header()
        if self.patience is not None:
            self.logger.info("Early stopping enabled: patience=%d epochs (monitoring val_acc)", self.patience)
        no_improve_epochs = 0
        stopped_early = False

        for epoch in range(1, self.epochs + 1):
            start_t = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_metrics, snr_breakdown = self.evaluate(self.val_loader)
            elapsed = time.time() - start_t

            current_val_acc = val_metrics["acc"]

            # Tie-breaking rule: Strictly greater keeps earliest epoch on ties
            is_best = current_val_acc > self.best_metric
            if is_best:
                self.best_metric = current_val_acc
                self.best_epoch = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "elapsed": elapsed,
                "is_best": is_best,
                "no_improve_epochs": no_improve_epochs,
            }
            self.history.append(epoch_record)

            # Log summary
            self.logger.info(
                "[Epoch %02d/%02d] (%.1fs) Train Loss: %.4f | Train Acc: %.2f%% | "
                "Val Loss: %.4f | Val Acc: %.2f%% | Prec: %.4f | Rec: %.4f | Macro-F1: %.4f | mAP: %.4f",
                epoch, self.epochs, elapsed, train_loss, train_acc * 100,
                val_loss, val_metrics["acc"] * 100, val_metrics["precision"],
                val_metrics["recall"], val_metrics["macro_f1"], val_metrics["mAP"]
            )

            # Log SNR Breakdown
            if snr_breakdown:
                self.logger.info("   -> SNR Breakdown (Epoch %02d):", epoch)
                for snr_name, stats in snr_breakdown.items():
                    self.logger.info(
                        "      SNR %6s: Acc=%.2f%% | Prec=%.4f | Rec=%.4f | Macro-F1=%.4f | mAP=%.4f (n=%d)",
                        snr_name, stats["acc"] * 100, stats["precision"], stats["recall"],
                        stats["macro_f1"], stats["mAP"], stats["n"]
                    )

            # Save state
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "config": self.config,
                "history": self.history,
            }
            save_checkpoint(state, self.checkpoint_dir, filename="last_model.pt", is_best=is_best)

            if is_best:
                self.logger.info(
                    "   [*] New best model saved to %s/best_model.pt (Val Acc: %.4f)",
                    self.checkpoint_dir, self.best_metric
                )
            elif self.patience is not None:
                self.logger.info(
                    "   [!] No val_acc improvement for %d/%d epochs (Best: %.2f%% at Epoch %d)",
                    no_improve_epochs, self.patience, self.best_metric * 100, self.best_epoch,
                )
                if no_improve_epochs >= self.patience:
                    stopped_early = True
                    self.logger.info(
                        "   [EARLY STOP] Triggered at Epoch %d (patience=%d). Best Epoch: %d (Val Acc: %.2f%%)",
                        epoch, self.patience, self.best_epoch, self.best_metric * 100,
                    )
                    break

        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETE: %s | Best Epoch: %d | Best Val Acc: %.2f%% | Early Stopped: %s",
                         self.model_name.upper(), self.best_epoch, self.best_metric * 100, stopped_early)
        self.logger.info("=" * 80)

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "stopped_early": stopped_early,
            "history": self.history,
        }
