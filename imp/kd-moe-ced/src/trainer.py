"""
Training and Optimization Engine for Single-Backbone MoE v2.

References:
    - Mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018. arXiv:1710.09412.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization", ACL 2024. arXiv:2401.06066.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_snr_breakdown


def apply_mixup(
    waveform: Tensor,
    target: Tensor,
    snr_db: Tensor,
    alpha: float = 0.4,
    prob: float = 0.8,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Apply Mixup data augmentation to waveforms, target labels, and continuous SNR targets.

    Reference:
        Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
    """
    if np.random.rand() > prob or alpha <= 0.0:
        return waveform, target, snr_db

    batch_size = waveform.shape[0]
    perm = torch.randperm(batch_size, device=waveform.device)
    lam = float(np.random.beta(alpha, alpha))

    mixed_wave = lam * waveform + (1.0 - lam) * waveform[perm]
    mixed_target = lam * target + (1.0 - lam) * target[perm]
    mixed_snr = lam * snr_db + (1.0 - lam) * snr_db[perm]

    return mixed_wave, mixed_target, mixed_snr


class Trainer:
    """
    Trainer supporting Single-Backbone MoE architectures, differential learning rates,
    dynamic soft gating supervision, and automatic validation.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 15,
        lr_backbone: float = 5.0e-5,
        lr_experts: float = 1.5e-4,
        lr_router: float = 2.0e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.8,
        label_smoothing: float = 0.05,
        lambda_route: float = 0.20,
        amp_dtype: str = "bfloat16",
        out_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.label_smoothing = label_smoothing
        self.lambda_route = lambda_route
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = device.type == "cuda"
        self.amp_type = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

        # Parameter groups with differential learning rates
        param_groups = self._build_param_groups(lr_backbone, lr_experts, lr_router, weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups)

        # Learning rate schedule: Linear Warmup + Cosine Decay
        total_steps = epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.best_macro_acc = 0.0
        self.best_results: dict[str, float] = {}

    def _build_param_groups(
        self, lr_backbone: float, lr_experts: float, lr_router: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        backbone_params, expert_params, router_params = [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name or "acoustic_extractor" in name:
                router_params.append(param)
            elif (
                "moe" in name
                or "hydra" in name
                or "lora" in name
                or "head_low" in name
                or "head_high" in name
                or "ffn_low" in name
                or "ffn_high" in name
                or "ffn_shared" in name
            ):
                expert_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        if expert_params:
            groups.append({"params": expert_params, "lr": lr_experts, "weight_decay": weight_decay})
        if router_params:
            groups.append({"params": router_params, "lr": lr_router, "weight_decay": weight_decay})

        # Fallback if no specific groups matched
        if not groups:
            groups.append({"params": list(self.model.parameters()), "lr": lr_backbone, "weight_decay": weight_decay})
        return groups

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, total_ce, total_route = 0.0, 0.0, 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            snr_db = batch["snr_db"].to(self.device, non_blocking=True)

            wave_m, target_m, snr_m = apply_mixup(
                wave, target, snr_db, alpha=self.mixup_alpha, prob=self.mixup_prob
            )

            # Label smoothing regularizer (Szegedy et al., CVPR 2016)
            if self.label_smoothing > 0.0:
                target_smooth = target_m * (1.0 - self.label_smoothing) + self.label_smoothing / 36.0
            else:
                target_smooth = target_m

            self.optimizer.zero_grad()
            with torch.autocast(self.device.type, dtype=self.amp_type, enabled=self.use_amp):
                out = self.model(wave_m, return_dict=True) if hasattr(self.model, "moe_blocks") or hasattr(self.model, "hydra_layers") else self.model(wave_m)

                if isinstance(out, dict):
                    logits_fused = out["logits"]
                    logits_low = out["logits_low"]
                    logits_high = out["logits_high"]
                    alpha = out["alpha"]

                    w_snr = torch.sigmoid((snr_m - 15.0) / 2.5).unsqueeze(-1)
                    log_probs_fused = F.log_softmax(logits_fused, dim=-1)
                    loss_ce = -(target_smooth * log_probs_fused).sum(dim=-1).mean()

                    log_probs_low = F.log_softmax(logits_low, dim=-1)
                    log_probs_high = F.log_softmax(logits_high, dim=-1)
                    loss_low = -(target_smooth * log_probs_low).sum(dim=-1)
                    loss_high = -(target_smooth * log_probs_high).sum(dim=-1)
                    loss_spec = ((1.0 - w_snr.squeeze(-1)) * loss_low + w_snr.squeeze(-1) * loss_high).mean()

                    with torch.autocast(self.device.type, enabled=False):
                        loss_route = F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), w_snr.float())
                    loss = loss_ce + 0.5 * loss_spec + self.lambda_route * loss_route.to(loss_ce.dtype)
                    loss_route_val = float(loss_route.item())
                else:
                    logits = out
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss_ce = -(target_m * log_probs).sum(dim=-1).mean()
                    loss = loss_ce
                    loss_route_val = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.item())
            total_ce += float(loss_ce.item())
            total_route += loss_route_val

            if (batch_idx + 1) % 60 == 0 or (batch_idx + 1) == num_batches:
                print(
                    f"  [Epoch {epoch + 1:02d} | Step {batch_idx + 1:03d}/{num_batches:03d}] "
                    f"Loss: {loss.item():.4f} | CE: {loss_ce.item():.4f} | "
                    f"Avg Loss: {total_loss / (batch_idx + 1):.4f}",
                    flush=True,
                )

        return {
            "loss": total_loss / num_batches,
            "loss_ce": total_ce / num_batches,
            "loss_route": total_route / num_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        all_probs, all_targets, all_snrs = [], [], []

        for batch in self.val_loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target_idx"].numpy()
            snr_db = batch["snr_db"].numpy()

            with torch.autocast(self.device.type, dtype=self.amp_type, enabled=self.use_amp):
                out = self.model(wave)
                logits = out["logits"] if isinstance(out, dict) else out
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target)
            all_snrs.append(snr_db)

        probs_arr = np.concatenate(all_probs, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        snrs_arr = np.concatenate(all_snrs, axis=0)

        return compute_snr_breakdown(probs_arr, targets_arr, snrs_arr, num_classes=36)

    def fit(self) -> dict[str, float]:
        print(f"Starting training for {self.epochs} epochs on device: {self.device}", flush=True)
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            macro_acc = val_metrics["macro_acc"]
            top1_acc = val_metrics["top1_accuracy"]
            map_score = val_metrics["mAP_macro"]

            print(
                f"[Epoch {epoch + 1:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Top-1 Acc: {top1_acc * 100:.2f}% | "
                f"Val Macro-Acc: {macro_acc * 100:.2f}% | "
                f"Val mAP: {map_score * 100:.2f}% | "
                f"20dB Acc: {val_metrics.get('acc_snr_20', 0) * 100:.2f}%",
                flush=True,
            )

            if macro_acc > self.best_macro_acc:
                self.best_macro_acc = macro_acc
                self.best_results = val_metrics
                torch.save(
                    {"model": self.model.state_dict(), "epoch": epoch + 1, "metrics": val_metrics},
                    self.out_dir / "best_model.pt",
                )
                print(f"--> Saved new best checkpoint (Macro-Acc: {macro_acc * 100:.2f}%)", flush=True)

        return self.best_results
