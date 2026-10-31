"""
Production Pipeline for Single-Backbone MoE v2 with Dual BEATs Teachers.

Supports:
1. Standalone Single-Backbone inference (CED-SingleMoE-SplitFFN / CED-HydraLoRA-MoE).
2. Self-Contained Champion Dynamic MoE Ensemble (Zero-Oracle SNR dependency):
   Dynamically blends student probabilities with BEATs Fold0 and BEATs EMA teachers
   using the student's internal Self-Contained Spectral-Token Router (alpha in [0, 1]).
3. 3-Way Temporal TTA (+/-50ms) with temperature calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.models.beat import BEATClassifier, load_fold_teacher
from src.models.moe_hydra import CEDHydraLoRAMoE
from src.models.moe_split import CEDSingleMoESplitFFN


def tta_forward(
    model: torch.nn.Module,
    wave: Tensor,
    device: torch.device,
    shift_samples: int = 0,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor | None]:
    """
    Forward pass with temporal shift.
    Returns (logits, alpha) where alpha is router weight if model outputs it.
    """
    if shift_samples != 0:
        if shift_samples > 0:
            w = F.pad(wave[:, shift_samples:], (0, shift_samples), mode="reflect")
        else:
            s = abs(shift_samples)
            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
    else:
        w = wave

    use_amp = device.type == "cuda"
    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        out = model(w, return_dict=True) if hasattr(model, "return_dict") or True else model(w)
        if isinstance(out, dict):
            logits = out["logits"].float()
            alpha = out.get("alpha", None)
            if alpha is not None:
                alpha = alpha.float()
        else:
            logits = out.float()
            alpha = None
    return logits, alpha


class ChampionPipelineV2:
    """
    Champion Pipeline v2 combining:
    - Single-Backbone Student (CED-SingleMoE-SplitFFN or CED-HydraLoRA-MoE)
    - Optional Dual-Teachers: BEATs Fold0 (clean-noise) & BEATs EMA (FRCRN-noise)
    - Self-Contained Spectral-Token Routing (alpha in [0, 1])
    - 3-Way Temporal TTA (+/-50ms)
    - Asymmetric Temperature Calibration (T_student=1.40, T_teacher=1.20)
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_fold0: torch.nn.Module | None = None,
        teacher_ema: torch.nn.Module | None = None,
        device: torch.device | None = None,
        student_temp: float = 1.40,
        teacher_temp: float = 1.20,
        student_weight: float = 0.25,
        t0_low: float = 0.45,
        tema_low: float = 0.30,
        t0_high: float = 0.30,
        tema_high: float = 0.45,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student_model.to(self.device).eval()
        self.t_fold0 = teacher_fold0.to(self.device).eval() if teacher_fold0 is not None else None
        self.t_ema = teacher_ema.to(self.device).eval() if teacher_ema is not None else None

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.student_weight = student_weight
        self.t0_low = t0_low
        self.tema_low = tema_low
        self.t0_high = t0_high
        self.tema_high = tema_high

    @torch.no_grad()
    def predict_student_tta(self, wave: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs 3-Way Temporal TTA on student model.
        Returns:
            probs: [B, num_classes] calibrated probabilities
            alphas: [B] router weights
        """
        wave = wave.to(self.device, non_blocking=True)
        l0, a0 = tta_forward(self.student, wave, self.device, shift_samples=0)
        lp, _ = tta_forward(self.student, wave, self.device, shift_samples=800)
        ln, _ = tta_forward(self.student, wave, self.device, shift_samples=-800)

        l_fused = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
        p_student = F.softmax(l_fused / self.student_temp, dim=-1).cpu().numpy()
        alphas = a0.squeeze().cpu().numpy() if a0 is not None else np.zeros(wave.shape[0])
        return p_student, alphas

    @torch.no_grad()
    def predict_proba(
        self,
        wave: Tensor,
        mode: Literal["standalone", "self_contained_champion", "oracle_champion"] = "self_contained_champion",
        snrs_db: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict class probabilities for batch of waveforms [B, 64000].
        """
        p_student, alphas = self.predict_student_tta(wave)

        if mode == "standalone" or self.t_fold0 is None or self.t_ema is None:
            return p_student

        # Teacher 3-Way TTA forward
        l0_t0, _ = tta_forward(self.t_fold0, wave, self.device, shift_samples=0)
        lp_t0, _ = tta_forward(self.t_fold0, wave, self.device, shift_samples=800)
        ln_t0, _ = tta_forward(self.t_fold0, wave, self.device, shift_samples=-800)
        l_t0 = (l0_t0 + 0.5 * lp_t0 + 0.5 * ln_t0) / 2.0
        p_t0 = F.softmax(l_t0 / self.teacher_temp, dim=-1).cpu().numpy()

        l0_te, _ = tta_forward(self.t_ema, wave, self.device, shift_samples=0)
        lp_te, _ = tta_forward(self.t_ema, wave, self.device, shift_samples=800)
        ln_te, _ = tta_forward(self.t_ema, wave, self.device, shift_samples=-800)
        l_te = (l0_te + 0.5 * lp_te + 0.5 * ln_te) / 2.0
        p_tema = F.softmax(l_te / self.teacher_temp, dim=-1).cpu().numpy()

        if mode == "oracle_champion":
            if snrs_db is None:
                raise ValueError("oracle_champion requires snrs_db array")
            is_high_snr = snrs_db >= 18.0
            p_final = np.zeros_like(p_student)
            p_final[~is_high_snr] = 0.35 * p_student[~is_high_snr] + 0.45 * p_t0[~is_high_snr] + 0.20 * p_tema[~is_high_snr]
            p_final[is_high_snr] = 0.35 * p_student[is_high_snr] + 0.40 * p_tema[is_high_snr] + 0.25 * p_t0[is_high_snr]
            return p_final

        # Self-contained dynamic soft blending via student router alpha:
        a = alphas[:, None] if alphas.ndim == 1 else alphas
        w_t0 = (1.0 - a) * self.t0_low + a * self.t0_high
        w_em = (1.0 - a) * self.tema_low + a * self.tema_high
        p_final = self.student_weight * p_student + w_t0 * p_t0 + w_em * p_tema
        return p_final

    def predict(
        self,
        wave: Tensor,
        mode: Literal["standalone", "self_contained_champion", "oracle_champion"] = "self_contained_champion",
        snrs_db: np.ndarray | None = None,
    ) -> np.ndarray:
        probs = self.predict_proba(wave, mode=mode, snrs_db=snrs_db)
        return np.argmax(probs, axis=-1)


def load_pipeline_v2(
    student_type: str = "split_ffn",
    student_ckpt: str | Path | None = None,
    teacher_fold0_ckpt: str | Path | None = None,
    teacher_ema_ckpt: str | Path | None = None,
    beat_config_path: str | Path | None = None,
    device: torch.device | None = None,
) -> ChampionPipelineV2:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if student_type.lower() in ("split_ffn", "splitffn", "moe_split"):
        student = CEDSingleMoESplitFFN(num_classes=36)
    elif student_type.lower() in ("hydra", "hydralora", "lora"):
        student = CEDHydraLoRAMoE(num_classes=36)
    else:
        raise ValueError(f"Unknown student_type: {student_type}")

    if student_ckpt is not None:
        ckpt_p = Path(student_ckpt)
        if ckpt_p.is_file():
            data = torch.load(ckpt_p, map_location=dev)
            state_dict = data["model"] if "model" in data else data
            student.load_state_dict(state_dict)
            print(f"Loaded student weights from {ckpt_p}")

    t_fold0 = None
    if teacher_fold0_ckpt is not None and Path(teacher_fold0_ckpt).is_file():
        cfg_beat = str(beat_config_path) if beat_config_path else "/marimo/moe-v2/config/beat.yaml"
        t_fold0, _ = load_fold_teacher(str(teacher_fold0_ckpt), cfg_beat, dev)
        print(f"Loaded Teacher Fold0 from {teacher_fold0_ckpt}")

    t_ema = None
    if teacher_ema_ckpt is not None and Path(teacher_ema_ckpt).is_file():
        cfg_beat = str(beat_config_path) if beat_config_path else "/marimo/moe-v2/config/beat.yaml"
        t_ema, _ = load_fold_teacher(str(teacher_ema_ckpt), cfg_beat, dev)
        print(f"Loaded Teacher EMA from {teacher_ema_ckpt}")

    return ChampionPipelineV2(
        student_model=student,
        teacher_fold0=t_fold0,
        teacher_ema=t_ema,
        device=dev,
        student_temp=1.40,
        teacher_temp=1.20,
        student_weight=0.25,
        t0_low=0.45,
        tema_low=0.30,
        t0_high=0.30,
        tema_high=0.45,
    )
