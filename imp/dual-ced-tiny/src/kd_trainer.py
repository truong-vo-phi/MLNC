"""
Logit-only Dual-Teacher Knowledge Distillation trainer for Dual-CED-Tiny.

Only the two CED-Tiny experts are trained. Each expert is distilled from its own frozen BEATs teacher through
softened logits (KL divergence); no feature / hidden / attention distillation is used.

  * Low expert  <- teacher logits [:, 0] (BEATs fold0)
  * High expert <- teacher logits [:, 1] (BEATs EMA)

  expert_loss = (1 - lambda_kd) * CE(z, y_smooth) + lambda_kd * T^2 * KL(softmax(t / T) || softmax(z / T))
  loss        = 0.5 * (expert_loss_low + expert_loss_high)

Validation / checkpoint selection (max val acc, earliest epoch on ties) is inherited from `Trainer`.
The self-contained spectral-token router receives no SNR metadata at validation or inference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.dataset import rms_normalize
from src.frontend import SpecAugment
from src.trainer import Trainer


class KDTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_logits_train: np.ndarray | Tensor,
        kd_config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        kd = kd_config or {}
        mix = kd.get("mixup", {}) or {}
        specaug_cfg = kd.get("specaugment", {}) or {}
        frontend_cfg = self.config.get("frontend", {}) or {}
        train_cfg = self.config.get("training", {}) or {}

        self.kd_temperature = float(kd.get("temperature", 2.0))
        self.lambda_kd = float(kd.get("lambda_kd", 0.65))
        self.lambda_route = float((self.config.get("router", {}) or {}).get("lambda_route_loss", 0.20))
        self.kd_label_smoothing = float(kd.get("label_smoothing", train_cfg.get("label_smoothing", 0.1)))
        self.input_rms = float(kd.get("input_rms", 0.07925))

        # Mixup 0.8 (enabled by default in KD)
        self.use_mixup = bool(mix.get("enabled", train_cfg.get("mixup", True)))
        self.mixup_alpha = float(mix.get("alpha", train_cfg.get("mixup_alpha", 0.4)))
        self.mixup_prob = float(mix.get("prob", train_cfg.get("mixup_prob", 0.8)))

        # SpecAugment (enabled by default in KD)
        self.use_specaugment = bool(specaug_cfg.get("enabled", train_cfg.get("specaugment", True)))
        freq_masks = int(specaug_cfg.get("freq_masks", frontend_cfg.get("specaug_freq_masks", 2)))
        freq_width = int(specaug_cfg.get("freq_width", frontend_cfg.get("specaug_freq_width", 8)))
        time_masks = int(specaug_cfg.get("time_masks", frontend_cfg.get("specaug_time_masks", 2)))
        time_width = int(specaug_cfg.get("time_width", frontend_cfg.get("specaug_time_width", 24)))
        self.spec_augment = SpecAugment(
            freq_masks=freq_masks if self.use_specaugment else 0,
            freq_width=freq_width,
            time_masks=time_masks if self.use_specaugment else 0,
            time_width=time_width,
        ).to(self.device)

        self.teacher_logits = torch.as_tensor(teacher_logits_train).float()  # [N, 2, C]
        if self.teacher_logits.ndim != 3 or self.teacher_logits.shape[1] != 2:
            raise ValueError(f"teacher_logits_train must be [N, 2, C], got {tuple(self.teacher_logits.shape)}")
        for name in ("low_expert", "high_expert", "frontend", "router"):
            if not hasattr(self.model, name):
                raise TypeError(f"KDTrainer requires a DualCEDTiny-like model (missing '{name}')")

    def _prepare_wave(self, wave: Tensor) -> Tensor:
        return rms_normalize(wave, target_rms=self.input_rms)

    def _log_header(self) -> None:
        super()._log_header()
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        n_low, n_high, n_all = count(self.model.low_expert), count(self.model.high_expert), count(self.model)
        self.logger.info("Only the 2 CED-Tiny experts are trained (BEATs teachers are frozen and outside the model).")
        self.logger.info("  Low expert : %s (%.2fM) <- logit KD from fold0 teacher", f"{n_low:,}", n_low / 1e6)
        self.logger.info("  High expert: %s (%.2fM) <- logit KD from EMA teacher", f"{n_high:,}", n_high / 1e6)
        n_router = count(self.model.router) + count(self.model.acoustic_extractor)
        self.logger.info("  Learned router + acoustic extractor: %s params | TOTAL: %s (%.2fM)",
                         f"{n_router:,}", f"{n_all:,}", n_all / 1e6)
        expert = self.model.low_expert
        alpha = (2 * len(expert.blocks)) ** 0.25
        self.logger.info(
            "  Student blocks: GREP=%s | Alpha Post-LN=%s | Pre-LN=%s | residual alpha=%.4f",
            expert.use_grep, expert.post_ln, expert.pre_ln, alpha,
        )
        self.logger.info(
            "  KD: fused/expert=%.2f/%.2f, T=%.2f, lambda_kd=%.2f, lambda_route=%.2f",
            float((self.config.get("kd", {}) or {}).get("fused_kd_weight", 0.65)),
            float((self.config.get("kd", {}) or {}).get("specialized_kd_weight", 0.35)),
            self.kd_temperature, self.lambda_kd, self.lambda_route,
        )
        self.logger.info(
            "  Augmentations: Mixup=%s(alpha=%.2f, prob=%.2f) | SpecAugment=%s(F=%dx%d, T=%dx%d)",
            self.use_mixup, self.mixup_alpha, self.mixup_prob,
            self.use_specaugment,
            self.spec_augment.freq_masks, self.spec_augment.freq_width,
            self.spec_augment.time_masks, self.spec_augment.time_width,
        )

    def _expert_loss(self, z: Tensor, target_smooth: Tensor, p_teacher: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ce = -(target_smooth * F.log_softmax(z, dim=-1)).sum(dim=-1).mean()
        # Match imp/moe-v2's soft-target cross-entropy distillation (T^2 scaled).
        kd = -(p_teacher * F.log_softmax(z / self.kd_temperature, dim=-1)).sum(dim=-1).mean()
        kd = kd * (self.kd_temperature ** 2)
        return (1.0 - self.lambda_kd) * ce + self.lambda_kd * kd, ce, kd

    def _dual_moe_loss(
        self,
        z_low: Tensor,
        z_high: Tensor,
        target_smooth: Tensor,
        p_teacher: Tensor,
        snr_db: Tensor,
        route_alpha: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """MoE-v2 fused/expert KD plus train-only soft-SNR router supervision."""
        kd_cfg = self.config.get("kd", {}) or {}
        fused_weight = float(kd_cfg.get("fused_kd_weight", 0.65))
        expert_weight = float(kd_cfg.get("specialized_kd_weight", 0.35))
        if fused_weight < 0 or expert_weight < 0 or abs(fused_weight + expert_weight - 1.0) > 1e-6:
            raise ValueError("fused_kd_weight and specialized_kd_weight must be nonnegative and sum to 1")

        temp = self.kd_temperature
        center = float(kd_cfg.get("snr_transition_center_db", 15.0))
        width = float(kd_cfg.get("snr_transition_tau_db", 2.5))
        w_high = torch.sigmoid((snr_db.float() - center) / width)
        p_teacher_blend = (1.0 - w_high[:, None]) * p_teacher[:, 0] + w_high[:, None] * p_teacher[:, 1]

        z_fused = (1.0 - route_alpha) * z_low + route_alpha * z_high
        ce = -(target_smooth * F.log_softmax(z_fused, dim=-1)).sum(dim=-1).mean()
        kd_fused = -(p_teacher_blend * F.log_softmax(z_fused / temp, dim=-1)).sum(dim=-1).mean() * (temp ** 2)

        kd_low = -(p_teacher[:, 0] * F.log_softmax(z_low / temp, dim=-1)).sum(dim=-1) * (temp ** 2)
        kd_high = -(p_teacher[:, 1] * F.log_softmax(z_high / temp, dim=-1)).sum(dim=-1) * (temp ** 2)
        kd_experts = ((1.0 - w_high) * kd_low + w_high * kd_high).mean()
        kd = fused_weight * kd_fused + expert_weight * kd_experts
        route_loss = self.model.router.compute_router_loss(route_alpha, snr_db)
        loss = (1.0 - self.lambda_kd) * ce + self.lambda_kd * kd + self.lambda_route * route_loss
        return loss, ce, kd, kd_experts, route_loss

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        self.spec_augment.train(self.use_specaugment)
        num_classes = self.model.num_classes
        sum_loss = sum_ce = sum_kd = sum_kd_experts = sum_route = 0.0
        correct = total = 0

        self.optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(self.train_loader):
            wave = self._prepare_wave(batch["waveform"].to(self.device, non_blocking=True))
            y = batch["target_idx"].to(self.device, non_blocking=True)
            snr = batch["snr_db"].to(self.device, non_blocking=True)
            tl = self.teacher_logits[batch["index"].long()].to(self.device)  # [B, 2, C]
            p_t = F.softmax(tl / self.kd_temperature, dim=-1)
            y1h = F.one_hot(y, num_classes).float()

            if self.use_mixup and np.random.rand() < self.mixup_prob:  # consistent mixup
                perm = torch.randperm(wave.size(0), device=self.device)
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                wave = lam * wave + (1.0 - lam) * wave[perm]
                y1h = lam * y1h + (1.0 - lam) * y1h[perm]
                p_t = lam * p_t + (1.0 - lam) * p_t[perm]
                snr = lam * snr + (1.0 - lam) * snr[perm]

            target_smooth = y1h * (1.0 - self.kd_label_smoothing) + self.kd_label_smoothing / num_classes

            spec = self.model.frontend(wave)  # fp32 STFT, outside autocast
            if self.use_specaugment:
                spec = self.spec_augment(spec)

            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                details = self.model.forward_from_features(wave, spec, return_details=True)
            z_low = details["logits_low"].float()
            z_high = details["logits_high"].float()
            route_alpha = details["alpha"].float()
            loss, ce, kd, kd_experts, route_loss = self._dual_moe_loss(
                z_low, z_high, target_smooth, p_t, snr, route_alpha
            )
            (loss / self.accumulation).backward()

            bs = wave.size(0)
            sum_loss += loss.item() * bs
            sum_ce += ce.item() * bs
            sum_kd += kd.item() * bs
            sum_kd_experts += kd_experts.item() * bs
            sum_route += route_loss.item() * bs
            fused = details["logits"].float()
            correct += (fused.argmax(-1) == y1h.argmax(-1)).sum().item()
            total += bs

            if (idx + 1) % self.accumulation == 0 or (idx + 1) == len(self.train_loader):
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        total = max(1, total)
        self.logger.info(
            "   -> Epoch %02d train: fused_CE=%.4f | KD_total=%.4f | KD_experts=%.4f | route_BCE=%.4f",
            epoch, sum_ce / total, sum_kd / total, sum_kd_experts / total, sum_route / total,
        )
        return sum_loss / total, correct / total

    @torch.no_grad()
    def _evaluate_snr_subset(self, loader: DataLoader, snr_db: float) -> dict[str, float]:
        """Evaluate learned-router predictions on the validation subset with SNR >= ``snr_db``."""
        self.model.eval()
        correct = total = 0
        for batch in loader:
            wave = self._prepare_wave(batch["waveform"].to(self.device, non_blocking=True))
            target_idx = batch["target_idx"].to(self.device, non_blocking=True)
            snr = batch["snr_db"].to(self.device, non_blocking=True)
            mask = snr >= snr_db
            if not bool(mask.any()):
                continue
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(wave, snr_db=snr)
            correct += int((logits.argmax(-1)[mask] == target_idx[mask]).sum().item())
            total += int(mask.sum().item())
        return {"acc": correct / max(1, total), "n": total}

    def finetune_high_expert(
        self,
        epochs: int = 6,
        lr: float = 5.0e-5,
        snr_db: float = 18.0,
        patience: int = 3,
    ) -> dict[str, Any]:
        """Stage-2: specialize the High expert using high-SNR training examples.

        Ground-truth SNR selects the Stage-2 training subset only. The learned router remains
        self-contained during validation. Only `high_expert` is optimized; the Low expert and
        router are frozen. The best epoch is selected on routed accuracy within the same SNR
        range, and `best_model.pt` is replaced only if overall routed validation accuracy improves.
        """
        dataset = self.train_loader.dataset
        snrs = getattr(dataset, "snrs", None)
        if snrs is None:
            self.logger.warning("High-expert fine-tune skipped: dataset exposes no '.snrs'.")
            return {"skipped": True, "reason": "dataset has no .snrs"}

        indices = np.flatnonzero(np.asarray(snrs) >= snr_db)
        if indices.size == 0:
            raise ValueError(f"No training clips with SNR >= {snr_db} dB; cannot specialize the High expert.")

        batch_size = int(getattr(self.train_loader, "batch_size", None) or 32)
        loader = DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Stage-2 requires the stage-1 checkpoint at {best_path}")
        stage1 = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(stage1["model_state_dict"])
        stage1_best = float(stage1.get("best_metric", 0.0))

        requires_grad_before = {name: p.requires_grad for name, p in self.model.named_parameters()}
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith("high_expert."))

        optimizer = torch.optim.AdamW(
            [p for p in self.model.high_expert.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        steps_per_epoch = max(1, math.ceil(len(loader) / self.accumulation))
        total_steps = steps_per_epoch * max(1, epochs)

        def lr_lambda(step: int) -> float:
            progress = min(1.0, float(step + 1) / float(total_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.logger.info("=" * 80)
        self.logger.info(
            "STAGE 2: High-expert specialization | SNR >= %.0f dB | clips=%d/%d | epochs=%d | lr=%g",
            snr_db, len(indices), len(dataset), epochs, lr,
        )
        self.logger.info("=" * 80)

        history: list[dict[str, Any]] = []
        best_snr_acc = -1.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.model.low_expert.eval()
            total_loss = total_samples = 0
            optimizer.zero_grad(set_to_none=True)

            for idx, batch in enumerate(loader):
                wave = self._prepare_wave(batch["waveform"].to(self.device, non_blocking=True))
                target_idx = batch["target_idx"].to(self.device, non_blocking=True)
                p_teacher = F.softmax(
                    self.teacher_logits[batch["index"].long()].to(self.device) / self.kd_temperature, dim=-1
                )[:, 1]
                target_smooth = F.one_hot(target_idx, self.model.num_classes).float()
                target_smooth = (
                    target_smooth * (1.0 - self.kd_label_smoothing) + self.kd_label_smoothing / self.model.num_classes
                )

                spec = self.model.frontend(wave)
                if self.use_specaugment:
                    spec = self.spec_augment(spec)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    logits = self.model.high_expert(spec)
                loss, _, _ = self._expert_loss(logits.float(), target_smooth, p_teacher)
                (loss / self.accumulation).backward()

                bs = wave.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
                if (idx + 1) % self.accumulation == 0 or (idx + 1) == len(loader):
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.high_expert.parameters(), max_norm=self.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            val_loss, metrics, _ = self.evaluate(self.val_loader)
            snr_acc = self._evaluate_snr_subset(self.val_loader, snr_db)["acc"]
            is_best = snr_acc > best_snr_acc
            if is_best:
                best_snr_acc = snr_acc
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": "high_ft",
                        "model_state_dict": self.model.state_dict(),
                        "best_snr_acc": best_snr_acc,
                        "snr_threshold_db": snr_db,
                        "fused_val_acc": metrics["acc"],
                    },
                    self.checkpoint_dir / "best_high_expert.pt",
                )
            else:
                no_improve += 1

            record = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_samples),
                "val_loss": val_loss,
                "val_snr_acc": snr_acc,
                "fused_val_acc": metrics["acc"],
                "is_best": is_best,
            }
            history.append(record)
            self.logger.info(
                "[High-FT %02d/%02d] Train Loss: %.4f | Val SNR>=%.0f Acc: %.2f%% | Fused Val Acc: %.2f%%%s",
                epoch, epochs, record["train_loss"], snr_db, snr_acc * 100, metrics["acc"] * 100,
                " [*BEST*]" if is_best else f" [Patience: {no_improve}/{patience}]",
            )
            if no_improve >= patience:
                self.logger.info("[High-FT] Early stop at epoch %d (patience=%d).", epoch, patience)
                break

        hf_ckpt = torch.load(self.checkpoint_dir / "best_high_expert.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(hf_ckpt["model_state_dict"])
        _, metrics, breakdown = self.evaluate(self.val_loader)
        adopted = metrics["acc"] > self.best_metric

        if adopted:
            self.best_metric = metrics["acc"]
            self.best_epoch = int(hf_ckpt["epoch"])
            torch.save(
                {
                    "epoch": self.best_epoch,
                    "stage": "high_ft",
                    "model_state_dict": self.model.state_dict(),
                    "best_metric": self.best_metric,
                    "best_epoch": self.best_epoch,
                    "val_metrics": metrics,
                    "snr_breakdown": breakdown,
                    "config": self.config,
                    "history": self.history,
                    "high_ft_history": history,
                },
                best_path,
            )
            self.logger.info(
                "[High-FT] ADOPTED: routed val acc %.2f%% (stage-1 best %.2f%%).",
                self.best_metric * 100, stage1_best * 100,
            )
        else:
            self.model.load_state_dict(stage1["model_state_dict"])
            self.logger.info(
                "[High-FT] NOT adopted: routed val acc %.2f%% <= stage-1 best %.2f%%.",
                metrics["acc"] * 100, self.best_metric * 100,
            )

        for name, p in self.model.named_parameters():
            p.requires_grad_(requires_grad_before[name])

        return {
            "enabled": True,
            "adopted": adopted,
            "best_epoch": int(hf_ckpt["epoch"]),
            "best_snr_acc": float(hf_ckpt["best_snr_acc"]),
            "fused_val_acc": float(metrics["acc"]),
            "history": history,
        }
