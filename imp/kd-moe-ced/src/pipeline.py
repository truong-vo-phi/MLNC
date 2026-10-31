"""
Production Pipeline for Single-Backbone MoE (CEDSingleMoESplitFFN) with Dual BEATs Teachers.

Supports:
1. Standalone Single-Backbone Student inference (Zero Oracle, ~6.62M params).
2. Dynamic Zero-Oracle Ensemble with Dual BEATs Teachers:
   - BEATs Fold0 (Clean/Environmental noise specialist)
   - BEATs EMA (FRCRN speech-leakage specialist)
   - Continuous soft interpolation using the internal router alpha in (0, 1).
3. 3-Way Temporal TTA (+/-50ms = +/-800 samples at 16 kHz).
4. Temperature Calibration (T_student = 1.40, T_teacher = 1.20).

References:
    - Soft MoE: Puigcerver et al., ICLR 2024.
    - BEATs: Chen et al., ICML 2023.
    - CED: Dinkel et al., ICASSP 2024.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.beat import BEATClassifier, load_fold_teacher
from src.models.split_moe import CEDSingleMoESplitFFN


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
        out = model(w, return_dict=True) if hasattr(model, "moe_blocks") else model(w)
        if isinstance(out, dict):
            logits = out["logits"].float()
            alpha = out.get("alpha", None)
            if alpha is not None:
                alpha = alpha.float()
        else:
            logits = out.float()
            alpha = None
    return logits, alpha


class ChampionPipeline:
    """
    Production Champion Pipeline combining:
    - Single-Backbone Student (CEDSingleMoESplitFFN)
    - Optional Dual-Teachers: BEATs Fold0 & BEATs EMA
    - Self-Contained Spectral-Token Router (alpha in [0, 1])
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
    def predict_student_tta(self, waveform: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Run 3-way temporal TTA (0, +50ms, -50ms) on student model.
        Returns (probs, alphas).
        """
        self.student.eval()
        w = waveform.to(self.device)
        shifts = [0, 800, -800]  # 50ms at 16kHz is 800 samples

        logits_list, alphas_list = [], []
        for s in shifts:
            l, a = tta_forward(self.student, w, self.device, shift_samples=s)
            logits_list.append(l)
            if a is not None:
                alphas_list.append(a)

        mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        p_student = F.softmax(mean_logits / self.student_temp, dim=-1).cpu().numpy()

        if alphas_list:
            mean_alpha = torch.stack(alphas_list, dim=0).mean(dim=0).cpu().numpy()
        else:
            mean_alpha = np.full((len(waveform), 1), 0.5)

        return p_student, mean_alpha

    @torch.no_grad()
    def predict_ensemble(
        self,
        waveform: Tensor,
        cached_t0_logits: Tensor | np.ndarray | None = None,
        cached_tema_logits: Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        End-to-End Zero-Oracle Dynamic Ensemble:
        Dynamically weights Teacher Fold0 and Teacher EMA according to router alpha.
        """
        p_student, alphas = self.predict_student_tta(waveform)

        # Obtain teacher 0 probabilities
        if cached_t0_logits is not None:
            if isinstance(cached_t0_logits, np.ndarray):
                cached_t0_logits = torch.from_numpy(cached_t0_logits)
            p_t0 = F.softmax(cached_t0_logits.float().to(self.device) / self.teacher_temp, dim=-1).cpu().numpy()
        elif self.t_fold0 is not None:
            l_t0, _ = tta_forward(self.t_fold0, waveform.to(self.device), self.device, shift_samples=0)
            p_t0 = F.softmax(l_t0 / self.teacher_temp, dim=-1).cpu().numpy()
        else:
            p_t0 = p_student

        # Obtain teacher EMA probabilities
        if cached_tema_logits is not None:
            if isinstance(cached_tema_logits, np.ndarray):
                cached_tema_logits = torch.from_numpy(cached_tema_logits)
            p_tema = F.softmax(cached_tema_logits.float().to(self.device) / self.teacher_temp, dim=-1).cpu().numpy()
        elif self.t_ema is not None:
            l_tema, _ = tta_forward(self.t_ema, waveform.to(self.device), self.device, shift_samples=0)
            p_tema = F.softmax(l_tema / self.teacher_temp, dim=-1).cpu().numpy()
        else:
            p_tema = p_student

        # Soft dynamic weight computation based on router alpha:
        # alpha -> 0: w_t0 = 0.45, w_tema = 0.30
        # alpha -> 1: w_t0 = 0.30, w_tema = 0.45
        w_t0 = (1.0 - alphas) * self.t0_low + alphas * self.t0_high
        w_tema = (1.0 - alphas) * self.tema_low + alphas * self.tema_high
        w_stu = self.student_weight

        # Normalize weights so they sum to 1
        total_w = w_stu + w_t0 + w_tema
        w_stu = w_stu / total_w
        w_t0 = w_t0 / total_w
        w_tema = w_tema / total_w

        p_final = w_stu * p_student + w_t0 * p_t0 + w_tema * p_tema
        return p_final, alphas


def load_pipeline(
    student_ckpt: Path | str | None = None,
    teacher_fold0_ckpt: Path | str | None = None,
    teacher_ema_ckpt: Path | str | None = None,
    beat_config_path: Path | str | None = None,
    device: torch.device | None = None,
) -> ChampionPipeline:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = CEDSingleMoESplitFFN(num_classes=36)
    if student_ckpt and Path(student_ckpt).is_file():
        ckpt = torch.load(student_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        student.load_state_dict(state, strict=False)
        print(f"Loaded student weights from {student_ckpt}")

    t_fold0 = None
    if teacher_fold0_ckpt and Path(teacher_fold0_ckpt).is_file() and beat_config_path:
        t_fold0 = load_fold_teacher(teacher_fold0_ckpt, beat_config_path, device=device)
        print(f"Loaded Teacher Fold0 from {teacher_fold0_ckpt}")

    t_ema = None
    if teacher_ema_ckpt and Path(teacher_ema_ckpt).is_file() and beat_config_path:
        t_ema = load_fold_teacher(teacher_ema_ckpt, beat_config_path, device=device)
        print(f"Loaded Teacher EMA from {teacher_ema_ckpt}")

    return ChampionPipeline(
        student_model=student,
        teacher_fold0=t_fold0,
        teacher_ema=t_ema,
        device=device,
    )
