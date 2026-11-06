"""
Training & Evaluation Engine for CED-SingleMoE-SplitFFN (~6.62M params).

Handles:
- Layer-wise differential learning rates (backbone, MoE experts, router).
- Linear warmup + Cosine Annealing scheduler.
- Multi-objective loss:
  1. Classification Loss (CrossEntropy / Label-Smoothed CE)
  2. Router Supervision Loss (BCE against continuous soft SNR target)
  3. Knowledge Distillation Loss (KL divergence from BEATs Teacher)
- Comprehensive evaluation on 9 Metrics:
  top1, top3, balanced_acc, map, auc, precision, recall, f1, mean_alpha.
- Detailed SNR breakdown across 6 levels: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB.
- Full .log recording and structured JSON summary export.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import compute_9_metrics_dict, compute_snr_breakdown
from src.models.ced_moe_split.model import CEDSingleMoESplitFFN


class MoETrainer:
    """
    Trainer for CED-SingleMoE-SplitFFN.
    """
    def __init__(
        self,
        student_model: CEDSingleMoESplitFFN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher_model: nn.Module | None = None,
        device: torch.device | str = "cpu",
        epochs: int = 15,
        lr_backbone: float = 5.0e-5,
        lr_experts: float = 1.5e-4,
        lr_router: float = 2.0e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        kd_temperature: float = 2.0,
        lambda_hard: float = 0.50,
        lambda_router: float = 0.20,
        lambda_kd: float = 0.50,
        checkpoint_dir: str | Path = "checkpoints/moe",
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.student = student_model.to(self.device)
        self.teacher = teacher_model.to(self.device) if teacher_model is not None else None
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.kd_temp = kd_temperature
        self.lambda_hard = lambda_hard
        self.lambda_router = lambda_router
        self.lambda_kd = lambda_kd if teacher_model is not None else 0.0

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("MoETrainer")

        # Layer-wise optimizer parameter groups
        backbone_params = []
        experts_params = []
        router_params = []

        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name:
                router_params.append(param)
            elif any(key in name for key in ("ffn_shared", "ffn_low", "ffn_high", "head_low", "head_high")):
                experts_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay, "initial_lr": lr_backbone},
                {"params": experts_params, "lr": lr_experts, "weight_decay": weight_decay, "initial_lr": lr_experts},
                {"params": router_params, "lr": lr_router, "weight_decay": 1e-4, "initial_lr": lr_router},
            ]
        )

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history: list[dict[str, Any]] = []

    def get_lr_scale(self, epoch: int) -> float:
        """Linear warmup followed by cosine annealing."""
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(max(1, self.warmup_epochs))
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.epochs - self.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.student.train()
        scale = self.get_lr_scale(epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * scale

        total_loss = 0.0
        ce_loss_sum = 0.0
        router_loss_sum = 0.0
        kd_loss_sum = 0.0
        correct_top1 = 0
        total_samples = 0

        for batch in self.train_loader:
            wave = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            snr_db = batch.get("snr_db", torch.zeros(len(wave))).to(self.device)

            self.optimizer.zero_grad()

            # Forward student
            out = self.student(wave, return_dict=True)
            logits = out["logits"]
            alpha = out["alpha"]

            # 1. Classification loss
            target_idx = torch.argmax(target, dim=-1) if target.ndim == 2 else target.long()
            loss_ce = self.ce_loss(logits, target_idx)

            # 2. Router loss
            loss_router = self.student.router.compute_router_loss(alpha, snr_db)

            # 3. Knowledge Distillation loss from BEATs
            loss_kd = torch.tensor(0.0, device=self.device)
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher(wave)
                p_t = F.softmax(teacher_logits / self.kd_temp, dim=-1)
                log_p_s = F.log_softmax(logits / self.kd_temp, dim=-1)
                loss_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.kd_temp ** 2)

            loss = (
                self.lambda_hard * loss_ce
                + self.lambda_router * loss_router
                + self.lambda_kd * loss_kd
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()

            bs = wave.size(0)
            total_loss += loss.item() * bs
            ce_loss_sum += loss_ce.item() * bs
            router_loss_sum += loss_router.item() * bs
            kd_loss_sum += loss_kd.item() * bs

            preds = torch.argmax(logits, dim=-1)
            correct_top1 += (preds == target_idx).sum().item()
            total_samples += bs

        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "loss_ce": ce_loss_sum / n,
            "loss_router": router_loss_sum / n,
            "loss_kd": kd_loss_sum / n,
            "acc": (correct_top1 / n) * 100.0,
        }

    @torch.no_grad()
    def evaluate(self) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        self.student.eval()
        all_logits = []
        all_targets = []
        all_snrs = []
        all_alphas = []
        total_loss = 0.0
        total_samples = 0

        for batch in self.val_loader:
            wave = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            snr_db = batch.get("snr_db", torch.zeros(len(wave))).to(self.device)

            out = self.student(wave, return_dict=True)
            logits = out["logits"]
            alpha = out["alpha"]

            target_idx = torch.argmax(target, dim=-1) if target.ndim == 2 else target.long()
            loss = self.ce_loss(logits, target_idx)

            bs = wave.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_logits.append(logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_snrs.append(snr_db.cpu().numpy())
            all_alphas.append(alpha.cpu().numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        snrs_arr = np.concatenate(all_snrs, axis=0)
        alphas_arr = np.concatenate(all_alphas, axis=0)

        metrics_9 = compute_9_metrics_dict(
            logits_arr, targets_arr, alphas=alphas_arr, num_classes=self.student.num_classes
        )
        metrics_9["val_loss"] = total_loss / max(total_samples, 1)

        breakdown = compute_snr_breakdown(
            logits_arr,
            targets_arr,
            snrs_arr,
            alphas=alphas_arr,
            num_classes=self.student.num_classes,
            use_9_metrics=True,
        )

        return metrics_9, breakdown

    def train(self) -> dict[str, Any]:
        params = self.student.get_num_params()
        self.logger.info("=" * 86)
        self.logger.info("TASK 4: PROPOSED CED-SINGLEMOE-SPLITFFN TRAINING & BENCHMARK")
        self.logger.info(f"Model Parameters: Total={params['total']/1e6:.2f}M | Trainable={params['trainable']/1e6:.2f}M | Router={params['router']:,}")
        self.logger.info(f"Hyperparameters: Epochs={self.epochs} | Warmup={self.warmup_epochs} | Device={self.device.type}")
        self.logger.info(f"Loss Weights: Lambda_Hard={self.lambda_hard} | Lambda_Router={self.lambda_router} | Lambda_KD={self.lambda_kd}")
        self.logger.info("9 Evaluation Metrics: top1, top3, balanced_acc, map, auc, precision, recall, f1, mean_alpha")
        self.logger.info("=" * 86)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_res = self.train_epoch(epoch - 1)
            val_res, breakdown = self.evaluate()
            elapsed = time.time() - t0

            # Primary tracking metric: top1 accuracy (%)
            curr_top1 = val_res["top1"] * 100.0 if val_res["top1"] <= 1.0 else val_res["top1"]
            is_best = curr_top1 > self.best_metric

            best_mark = " [* BEST MODEL]" if is_best else ""
            self.logger.info(
                f"[Epoch {epoch:02d}/{self.epochs:02d}] ({elapsed:.1f}s) "
                f"Train Loss: {train_res['loss']:.4f} (CE: {train_res['loss_ce']:.4f}, R: {train_res['loss_router']:.4f}) | "
                f"Val Loss: {val_res['val_loss']:.4f} | "
                f"Top-1: {curr_top1:.2f}% | Top-3: {val_res['top3']*100:.2f}% | "
                f"BalAcc: {val_res['balanced_acc']*100:.2f}% | mAP: {val_res['map']:.4f} | "
                f"AUC: {val_res['auc']:.4f} | F1: {val_res['f1']:.4f} | "
                f"Mean Alpha: {val_res['mean_alpha']:.4f}{best_mark}"
            )

            # Detailed 9 Metrics SNR Breakdown Logging
            self.logger.info(f"  -> SNR Breakdown (Epoch {epoch:02d}):")
            for snr_key in ("-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"):
                if snr_key in breakdown:
                    row = breakdown[snr_key]
                    cnt = int(row.get("count", 0))
                    t1 = row.get("top1", 0.0) * 100.0 if row.get("top1", 0.0) <= 1.0 else row.get("top1", 0.0)
                    t3 = row.get("top3", 0.0) * 100.0 if row.get("top3", 0.0) <= 1.0 else row.get("top3", 0.0)
                    mp = row.get("map", 0.0)
                    f1 = row.get("f1", 0.0)
                    al = row.get("mean_alpha", 0.50)
                    self.logger.info(
                        f"     SNR {snr_key:>5s}: Top-1={t1:5.2f}% | Top-3={t3:5.2f}% | "
                        f"mAP={mp:.4f} | F1={f1:.4f} | Alpha={al:.4f} (n={cnt})"
                    )

            if is_best or epoch == 1:
                self.best_metric = curr_top1
                self.best_epoch = epoch
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.student.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_res,
                        "breakdown": breakdown,
                    },
                    ckpt_path,
                )
                self.logger.info(f"  [*] Saved new best model to {ckpt_path} (Top-1: {curr_top1:.2f}%)")

            self.history.append({
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "val_loss": val_res["val_loss"],
                "top1_acc": curr_top1,
                "val_metrics": val_res,
                "breakdown": breakdown,
                "is_best": is_best,
            })

        total_min = (time.time() - start_time) / 60.0
        self.logger.info("=" * 86)
        self.logger.info(f"FINISHED TASK 4 BENCHMARK in {total_min:.1f} minutes")
        self.logger.info(f"Best Validation Top-1 Acc: {self.best_metric:.2f}% at Epoch {self.best_epoch}")
        self.logger.info("=" * 86)

        return {
            "best_epoch": self.best_epoch,
            "best_top1_acc": self.best_metric,
            "history": self.history,
        }

