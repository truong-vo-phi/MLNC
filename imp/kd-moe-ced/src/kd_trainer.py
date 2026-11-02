"""
Động cơ Huấn luyện Chưng cất Tri thức (Knowledge Distillation Engine) cho Single-Backbone MoE (CEDSingleMoESplitFFN).
Thực hiện chưng cất tri thức kép (Dual-Teacher Distillation) từ BEATs-12L (Fold0 Clean/Noise & EMA FRCRN).

Tài liệu tham khảo:
    - KD: Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014 Workshop.
    - Consistent Distillation: Beyer et al., "Knowledge Distillation: A Good Teacher is Patient and Consistent", CVPR 2022.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization", ACL 2024.
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


def apply_mixup_with_teachers(
    waveform: Tensor,
    target: Tensor,
    snr_db: Tensor,
    t0_logits: Tensor,
    tema_logits: Tensor,
    alpha: float = 0.4,
    prob: float = 0.8,
    temp: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Áp dụng kỹ thuật tăng cường Consistent Mixup đồng thời trên:
    - Dạng sóng âm thanh (waveforms)
    - Nhãn mục tiêu ground-truth (targets)
    - Giá trị SNR liên tục (snr_db)
    - Phân phối xác suất mềm của cả 2 giáo viên (Beyer et al., CVPR 2022)
    """
    p_t0 = F.softmax(t0_logits / temp, dim=-1)
    p_tema = F.softmax(tema_logits / temp, dim=-1)

    if np.random.rand() > prob or alpha <= 0.0:
        return waveform, target, snr_db, p_t0, p_tema

    batch_size = waveform.shape[0]
    perm = torch.randperm(batch_size, device=waveform.device)
    lam = float(np.random.beta(alpha, alpha))

    mixed_wave = lam * waveform + (1.0 - lam) * waveform[perm]
    mixed_target = lam * target + (1.0 - lam) * target[perm]
    mixed_snr = lam * snr_db + (1.0 - lam) * snr_db[perm]
    mixed_pt0 = lam * p_t0 + (1.0 - lam) * p_t0[perm]
    mixed_ptema = lam * p_tema + (1.0 - lam) * p_tema[perm]

    return mixed_wave, mixed_target, mixed_snr, mixed_pt0, mixed_ptema


class KDTrainer:
    """
    Trình quản lý Huấn luyện Chưng cất Tri thức Kép cho Single-Backbone MoE.
    Tích hợp hòa trộn mềm giáo viên, chưng cất chuyên biệt hóa chuyên gia,
    giám sát bộ định tuyến động, và lịch trình suy giảm tốc độ học hình cosine (Cosine Annealing).
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 15,
        lr_backbone: float = 4.0e-5,
        lr_experts: float = 1.2e-4,
        lr_router: float = 2.0e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.8,
        kd_temperature: float = 2.0,
        lambda_hard: float = 0.20,
        lambda_kd: float = 0.70,
        lambda_router: float = 0.10,
        out_dir: Path | str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.kd_temp = kd_temperature
        self.lambda_hard = lambda_hard
        self.lambda_kd = lambda_kd
        self.lambda_router = lambda_router
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_top1_acc = 0.0
        self.best_results: dict[str, Any] = {}

        # Phân tách nhóm tham số và gán tốc độ học phân tầng (Layer-wise LR)
        backbone_params = []
        experts_params = []
        router_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name:
                router_params.append(param)
            elif "moe_blocks" in name or "head_low" in name or "head_high" in name:
                experts_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": experts_params, "lr": lr_experts, "weight_decay": weight_decay},
                {"params": router_params, "lr": lr_router, "weight_decay": 1e-4},
            ]
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def get_lr_scale(self, epoch: int) -> float:
        """Hệ số điều chỉnh tốc độ học theo Cosine Annealing kèm Warmup tuyến tính."""
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(max(1, self.warmup_epochs))
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.epochs - self.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Thực hiện một epoch huấn luyện."""
        self.model.train()
        scale = self.get_lr_scale(epoch)
        for pg in self.optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]
            pg["lr"] = pg["initial_lr"] * scale

        total_loss = 0.0
        loss_h_sum, loss_kd_sum, loss_r_sum = 0.0, 0.0, 0.0
        batches = 0

        for batch in self.train_loader:
            wave = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            snr_db = batch["snr_db"].to(self.device)
            t0_logits = batch.get("t0_logits", torch.zeros_like(target)).to(self.device)
            tema_logits = batch.get("tema_logits", torch.zeros_like(target)).to(self.device)

            # Áp dụng Consistent Mixup
            wave, target, snr_db, p_t0, p_tema = apply_mixup_with_teachers(
                wave, target, snr_db, t0_logits, tema_logits,
                alpha=self.mixup_alpha, prob=self.mixup_prob, temp=self.kd_temp
            )

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                out = self.model(wave, return_dict=True)
                logits = out["logits"]
                logits_low = out["logits_low"]
                logits_high = out["logits_high"]
                alpha = out["alpha"]

                # 1. Hàm mất mát nhãn gốc Hard Label Loss (BCEWithLogits)
                loss_hard = self.bce_loss(logits, target)

                # 2. Phân phối mềm mục tiêu kết hợp từ hai giáo viên:
                # y_target = Sigmoid((SNR - 15) / 2.5)
                # target_teacher = (1 - y_target) * p_t0 + y_target * p_tema
                y_target = self.model.router.compute_target_soft_label(snr_db)
                p_teacher_blended = (1.0 - y_target) * p_t0 + y_target * p_tema

                # Chưng cất KL Divergence cho mô hình tổng thể
                log_p_student = F.log_softmax(logits / self.kd_temp, dim=-1)
                loss_kd_global = F.kl_div(log_p_student, p_teacher_blended, reduction="batchmean") * (self.kd_temp ** 2)

                # Chưng cất chuyên biệt hóa từng chuyên gia (Expert Specialization Distillation)
                log_p_low = F.log_softmax(logits_low / self.kd_temp, dim=-1)
                log_p_high = F.log_softmax(logits_high / self.kd_temp, dim=-1)
                loss_kd_low = F.kl_div(log_p_low, p_t0, reduction="batchmean") * (self.kd_temp ** 2)
                loss_kd_high = F.kl_div(log_p_high, p_tema, reduction="batchmean") * (self.kd_temp ** 2)
                loss_kd = loss_kd_global + 0.5 * (loss_kd_low + loss_kd_high)

                # 3. Mất mát giám sát bộ định tuyến (Router Supervision Loss)
                loss_router = self.model.router.compute_router_loss(alpha, snr_db)

                # Tổng hợp hàm mất mát
                loss = (
                    self.lambda_hard * loss_hard
                    + self.lambda_kd * loss_kd
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
            loss_r_sum += float(loss_router.item())
            batches += 1

        n = max(batches, 1)
        return {
            "loss": total_loss / n,
            "loss_hard": loss_h_sum / n,
            "loss_kd": loss_kd_sum / n,
            "loss_router": loss_r_sum / n,
            "lr_backbone": self.optimizer.param_groups[0]["lr"],
            "lr_experts": self.optimizer.param_groups[1]["lr"],
            "lr_router": self.optimizer.param_groups[2]["lr"],
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, dict[str, float]]:
        """Đánh giá mô hình trên tập kiểm tra val_loader."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_snrs = []

        for batch in self.val_loader:
            wave = batch["waveform"].to(self.device)
            target = batch["target_idx"].numpy()
            snr = batch["snr_db"].numpy()

            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                logits = self.model(wave, return_dict=False)
                probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(target)
            all_snrs.append(snr)

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        snrs = np.concatenate(all_snrs, axis=0)

        return compute_snr_breakdown(preds, targets, snrs)

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Lưu checkpoint mô hình."""
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
