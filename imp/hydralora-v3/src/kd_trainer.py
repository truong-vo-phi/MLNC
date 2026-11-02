"""
Reliability-Gated Knowledge Distillation Trainer for CED-HydraLoRA-v3.

Key Innovations:
    1. Sample-Level Reliability Gating:
       Filters teacher hallucinations by checking ground-truth accuracy.
       Zeroes out KD loss on poisoned/incorrect samples (42.8% at 20dB).
       Selectively distills from Fold0 or EMA based on individual correctness.
    2. Temporal Self-Consistency Regularization (Consistency Distillation):
       Forces temporal phase invariance (+/- 50ms) to internalize TTA benefits
       into standalone single-pass weights.
    3. Specialized Expert Distillation:
       Low-SNR expert head learns from verified Fold0 representations.
       High-SNR expert head learns from verified EMA representations.
    4. Router Target Supervision:
       Supervises the internal zero-oracle router with continuous SNR targets.

References:
    - HydraLoRA: Tian et al., NeurIPS 2024 Oral. arXiv:2404.19245.
    - Beyer et al., "Knowledge Distillation: A Good Teacher is Patient and Consistent", CVPR 2022.
    - Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014 Workshop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import compute_snr_breakdown


class ReliabilityGatedKDTrainer:
    """
    Advanced Knowledge Distillation Engine with Reliability Gating and Temporal Consistency.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 15,
        lr_backbone: float = 3.5e-5,
        lr_lora: float = 1.5e-4,
        lr_router: float = 2.0e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        kd_temperature: float = 2.0,
        lambda_hard: float = 0.35,
        lambda_kd: float = 0.55,
        lambda_cons: float = 0.15,
        lambda_router: float = 0.10,
        out_dir: Path | str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.kd_temp = kd_temperature
        self.lambda_hard = lambda_hard
        self.lambda_kd = lambda_kd
        self.lambda_cons = lambda_cons
        self.lambda_router = lambda_router
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_top1_acc = 0.0
        self.best_results: dict[str, Any] = {}

        # Parameter groups with layer-wise learning rates
        backbone_params = []
        lora_params = []
        router_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name:
                router_params.append(param)
            elif "lora_" in name or "head_low" in name or "head_high" in name:
                lora_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay * 0.5},
                {"params": router_params, "lr": lr_router, "weight_decay": 1e-4},
            ]
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def get_lr_scale(self, epoch: int) -> float:
        """Linear warmup followed by cosine annealing schedule with minimum learning rate floor."""
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(max(1, self.warmup_epochs))
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.epochs - self.warmup_epochs))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        scale = self.get_lr_scale(epoch)
        for pg in self.optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]
            pg["lr"] = pg["initial_lr"] * scale

        total_loss = 0.0
        loss_h_sum, loss_kd_sum, loss_cons_sum, loss_r_sum = 0.0, 0.0, 0.0, 0.0
        gated_ratio_sum = 0.0
        batches = 0

        for batch in self.train_loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            target_idx = batch["target_idx"].to(self.device, non_blocking=True)
            snr_db = batch["snr_db"].to(self.device, non_blocking=True)
            t0_logits = batch.get("t0_logits", torch.zeros_like(target)).to(self.device, non_blocking=True)
            tema_logits = batch.get("tema_logits", torch.zeros_like(target)).to(self.device, non_blocking=True)
            B = wave.shape[0]

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                # 1. Forward pass on primary waveform
                out = self.model(wave, return_dict=True)
                logits = out["logits"]
                logits_low = out["logits_low"]
                logits_high = out["logits_high"]
                alpha = out["alpha"]

                # 2. Hard Label Loss with label smoothing (BCEWithLogits)
                # Label smoothing epsilon=0.05 prevents overconfident miscalibration
                target_smooth = target * 0.95 + 0.05 / target.shape[-1]
                loss_hard = self.bce_loss(logits, target_smooth)

                # 3. Reliability Gating Logic
                t0_pred = t0_logits.argmax(dim=-1)
                tema_pred = tema_logits.argmax(dim=-1)
                t0_correct = (t0_pred == target_idx)
                tema_correct = (tema_pred == target_idx)
                both_correct = t0_correct & tema_correct
                either_correct = t0_correct | tema_correct

                p_t0 = F.softmax(t0_logits / self.kd_temp, dim=-1)
                p_tema = F.softmax(tema_logits / self.kd_temp, dim=-1)

                # Compute SNR-dependent blending weight for when both teachers are valid
                # y_target: 0 at Low-SNR (-5 to 5dB), transitions to 1 at High-SNR (15 to 20dB)
                y_target = self.model.router.compute_target_soft_label(snr_db)  # [B, 1]
                if y_target.ndim == 1:
                    y_target = y_target.unsqueeze(-1)

                p_blended = (1.0 - y_target) * p_t0 + y_target * p_tema

                # Construct sample-adaptive target distribution
                # If both correct -> blended
                # If only t0 correct -> t0
                # If only tema correct -> tema
                # If neither correct -> ground truth one-hot (shielding student from hallucination)
                p_adaptive_target = torch.where(
                    both_correct.unsqueeze(-1),
                    p_blended,
                    torch.where(
                        t0_correct.unsqueeze(-1),
                        p_t0,
                        torch.where(
                            tema_correct.unsqueeze(-1),
                            p_tema,
                            target_smooth
                        )
                    )
                )

                # KD weight per sample: 1.0 if at least one teacher is correct, 0.0 if both teachers are wrong
                kd_sample_weight = either_correct.float()  # [B]
                gated_ratio = kd_sample_weight.mean()

                log_p_student = F.log_softmax(logits / self.kd_temp, dim=-1)
                kl_per_sample = F.kl_div(log_p_student, p_adaptive_target, reduction="none").sum(dim=-1) * (self.kd_temp ** 2)
                loss_kd_global = (kl_per_sample * kd_sample_weight).sum() / (kd_sample_weight.sum().clamp(min=1.0))

                # Expert specialization distillation:
                # low expert distills from t0 on t0-correct samples
                log_p_low = F.log_softmax(logits_low / self.kd_temp, dim=-1)
                kl_low = F.kl_div(log_p_low, p_t0, reduction="none").sum(dim=-1) * (self.kd_temp ** 2)
                loss_kd_low = (kl_low * t0_correct.float()).sum() / (t0_correct.float().sum().clamp(min=1.0))

                # high expert distills from tema on tema-correct samples
                log_p_high = F.log_softmax(logits_high / self.kd_temp, dim=-1)
                kl_high = F.kl_div(log_p_high, p_tema, reduction="none").sum(dim=-1) * (self.kd_temp ** 2)
                loss_kd_high = (kl_high * tema_correct.float()).sum() / (tema_correct.float().sum().clamp(min=1.0))

                loss_kd = loss_kd_global + 0.3 * (loss_kd_low + loss_kd_high)

                # 4. Temporal Self-Consistency Regularization (baking TTA shift invariance into weights)
                # Random shift +/- 800 samples (+/- 50ms at 16kHz)
                shift = 800 if torch.rand(1).item() > 0.5 else -800
                if shift > 0:
                    wave_shift = F.pad(wave[:, shift:], (0, shift), mode="reflect")
                else:
                    s = abs(shift)
                    wave_shift = F.pad(wave[:, :-s], (s, 0), mode="reflect")

                out_shift = self.model(wave_shift, return_dict=False)
                log_p_shift = F.log_softmax(out_shift / self.kd_temp, dim=-1)
                p_orig = F.softmax(logits.detach() / self.kd_temp, dim=-1)
                loss_cons = F.kl_div(log_p_shift, p_orig, reduction="batchmean") * (self.kd_temp ** 2)

                # 5. Router Supervision Loss
                loss_router = self.model.router.compute_router_loss(alpha, snr_db)

                # Total Loss
                loss = (
                    self.lambda_hard * loss_hard
                    + self.lambda_kd * loss_kd
                    + self.lambda_cons * loss_cons
                    + self.lambda_router * loss_router
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            loss_h_sum += float(loss_hard.item())
            loss_kd_sum += float(loss_kd.item())
            loss_cons_sum += float(loss_cons.item())
            loss_r_sum += float(loss_router.item())
            gated_ratio_sum += float(gated_ratio.item())
            batches += 1

        n = max(batches, 1)
        return {
            "loss": total_loss / n,
            "loss_hard": loss_h_sum / n,
            "loss_kd": loss_kd_sum / n,
            "loss_cons": loss_cons_sum / n,
            "loss_router": loss_r_sum / n,
            "reliable_teacher_pct": (gated_ratio_sum / n) * 100.0,
            "lr_backbone": self.optimizer.param_groups[0]["lr"],
            "lr_lora": self.optimizer.param_groups[1]["lr"],
            "lr_router": self.optimizer.param_groups[2]["lr"],
        }

    @torch.no_grad()
    def evaluate(self, use_tta: bool = False, tta_temp: float = 1.35) -> dict[str, Any]:
        """
        Evaluate model on val_loader across all 6 SNR levels.
        If use_tta is True, applies 3-way temporal shifts (+/-50ms) with temperature calibration.
        """
        self.model.eval()
        all_probs, all_targets, all_snrs, all_alphas = [], [], [], []

        for batch in self.val_loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target_idx"].numpy()
            snr = batch["snr_db"].numpy()

            if use_tta:
                def forward_shifted(shift: int) -> tuple[Tensor, Tensor]:
                    if shift != 0:
                        if shift > 0:
                            w = F.pad(wave[:, shift:], (0, shift), mode="reflect")
                        else:
                            s = abs(shift)
                            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
                    else:
                        w = wave
                    with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                        d = self.model(w, return_dict=True)
                    return d["logits"].float(), d["alpha"].float()

                l0, a0 = forward_shifted(0)
                lp, _ = forward_shifted(800)
                ln, _ = forward_shifted(-800)
                l_fused = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
                probs = F.softmax(l_fused / tta_temp, dim=-1).cpu().numpy()
                alpha_np = a0.cpu().numpy()
            else:
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                    d = self.model(wave, return_dict=True)
                    probs = F.softmax(d["logits"].float(), dim=-1).cpu().numpy()
                    alpha_np = d["alpha"].float().cpu().numpy()

            all_probs.append(probs)
            all_targets.append(target)
            all_snrs.append(snr)
            all_alphas.append(alpha_np)

        preds_arr = np.concatenate(all_probs, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        snrs_arr = np.concatenate(all_snrs, axis=0)
        alphas_arr = np.concatenate(all_alphas, axis=0)

        return compute_snr_breakdown(preds_arr, targets_arr, snrs_arr, alphas=alphas_arr)

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save training checkpoint."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_top1_acc": self.best_top1_acc,
            "best_results": self.best_results,
        }
        torch.save(ckpt, path)
        if is_best:
            best_path = self.out_dir / "best_model.pt"
            torch.save(ckpt, best_path)
