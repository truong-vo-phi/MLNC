"""
Pipeline Triển khai Thực tế (Production Pipeline) cho CEDSingleMoESplitFFN kết hợp Dual BEATs Teachers.

Hỗ trợ các tính năng:
1. Suy luận Độc lập (Standalone Single-Backbone Student): Tự thân 100% Zero-Oracle (~6.62M tham số).
2. Kết hợp Động (Dynamic Zero-Oracle Ensemble) với hai giáo viên BEATs:
   - BEATs Fold0 (Chuyên gia nhận diện nhiễu môi trường SNR thấp).
   - BEATs EMA (Chuyên gia khử rò rỉ tiếng nói FRCRN ở SNR cao).
   - Nội suy mềm liên tục dựa trên trọng số alpha thuộc (0, 1) do bộ định tuyến nội tại sinh ra.
3. Tăng cường thời điểm kiểm tra 3-Way Temporal TTA (+/-50ms = +/-800 mẫu ở 16 kHz).
4. Hiệu chuẩn Nhiệt độ Bất đối xứng (Temperature Calibration: T_student = 1.40, T_teacher = 1.20).

Tài liệu tham khảo:
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

try:
    from src.models.beat import BEATClassifier, load_fold_teacher
except Exception:
    BEATClassifier = None
    load_fold_teacher = None
from src.models.split_moe import CEDSingleMoESplitFFN



def tta_forward(
    model: torch.nn.Module,
    wave: Tensor,
    device: torch.device,
    shift_samples: int = 0,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor | None]:
    """
    Thực hiện forward pass kèm dịch chuyển thời gian (temporal shift).
    Trả về (logits, alpha) với alpha là trọng số định tuyến nếu mô hình hỗ trợ.
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
    Champion Pipeline kết hợp hoàn chỉnh:
    - Mô hình học sinh độc lập (CEDSingleMoESplitFFN)
    - Tùy chọn hai giáo viên (BEATs Fold0 & BEATs EMA)
    - Bộ định tuyến Phổ - Token Tự thân (tính alpha thuộc [0, 1])
    - Tăng cường thời điểm kiểm tra 3-Way Temporal TTA (+/-50ms)
    - Hiệu chuẩn nhiệt độ bất đối xứng (T_student=1.40, T_teacher=1.20)
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_fold0: torch.nn.Module | None = None,
        teacher_ema: torch.nn.Module | None = None,
        device: torch.device | None = None,
        t_student: float = 1.40,
        t_teacher: float = 1.20,
        w_student: float = 0.25,
        t0_low: float = 0.45,
        tema_low: float = 0.30,
        t0_high: float = 0.30,
        tema_high: float = 0.45,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student_model.to(self.device).eval()
        self.teacher_fold0 = teacher_fold0.to(self.device).eval() if teacher_fold0 else None
        self.teacher_ema = teacher_ema.to(self.device).eval() if teacher_ema else None
        self.t_student = t_student
        self.t_teacher = t_teacher
        self.w_student = w_student
        self.t0_low = t0_low
        self.tema_low = tema_low
        self.t0_high = t0_high
        self.tema_high = tema_high

    @torch.no_grad()
    def predict_student_tta(self, waveform: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán đầu ra của Student với kỹ thuật 3-Way Temporal TTA (0, +50ms, -50ms).
        Trả về (probs_student, alphas).
        """
        shifts = [0, 800, -800]  # 800 mẫu tương ứng với 50ms tại tần số lấy mẫu 16 kHz
        all_logits = []
        all_alphas = []

        for shift in shifts:
            logits, alpha = tta_forward(self.student, waveform, self.device, shift_samples=shift)
            all_logits.append(logits)
            if alpha is not None:
                all_alphas.append(alpha)

        # Lấy trung bình cộng logits qua 3 phép dịch chuyển
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        # Áp dụng nhiệt độ hiệu chuẩn học sinh
        probs = F.softmax(avg_logits / self.t_student, dim=-1).cpu().numpy()

        if all_alphas:
            avg_alpha = torch.stack(all_alphas, dim=0).mean(dim=0).cpu().numpy()
        else:
            avg_alpha = np.full((len(waveform), 1), 0.5, dtype=np.float32)

        return probs, avg_alpha

    @torch.no_grad()
    def predict_teacher_blended(
        self,
        waveform: Tensor,
        alpha: np.ndarray,
        cached_t0_logits: np.ndarray | None = None,
        cached_tema_logits: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Nội suy mềm phân phối xác suất của hai giáo viên theo trọng số alpha:
        p_teacher = (1 - alpha) * p_fold0 + alpha * p_ema
        """
        if cached_t0_logits is not None and cached_tema_logits is not None:
            l_t0 = torch.from_numpy(cached_t0_logits).to(self.device).float()
            l_tema = torch.from_numpy(cached_tema_logits).to(self.device).float()
            p_t0 = F.softmax(l_t0 / self.t_teacher, dim=-1).cpu().numpy()
            p_tema = F.softmax(l_tema / self.t_teacher, dim=-1).cpu().numpy()
        else:
            assert self.teacher_fold0 is not None and self.teacher_ema is not None
            shifts = [0, 800, -800]
            l_t0_list, l_tema_list = [], []
            for s in shifts:
                l0, _ = tta_forward(self.teacher_fold0, waveform, self.device, shift_samples=s)
                lema, _ = tta_forward(self.teacher_ema, waveform, self.device, shift_samples=s)
                l_t0_list.append(l0)
                l_tema_list.append(lema)
            l_t0 = torch.stack(l_t0_list, dim=0).mean(dim=0)
            l_tema = torch.stack(l_tema_list, dim=0).mean(dim=0)
            p_t0 = F.softmax(l_t0 / self.t_teacher, dim=-1).cpu().numpy()
            p_tema = F.softmax(l_tema / self.t_teacher, dim=-1).cpu().numpy()

        a = alpha.reshape(-1, 1)
        w_t0 = (1.0 - a) * self.t0_low + a * self.t0_high
        w_tema = (1.0 - a) * self.tema_low + a * self.tema_high
        return w_t0 * p_t0 + w_tema * p_tema

    @torch.no_grad()
    def predict_ensemble(
        self,
        waveform: Tensor,
        cached_t0_logits: np.ndarray | None = None,
        cached_tema_logits: np.ndarray | None = None,
        p_student: np.ndarray | None = None,
        alphas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán tổng hợp Ensemble Zero-Oracle kết hợp Student và Dual-Teachers:
        p_final = w_student * p_student + (1 - w_student) * p_teachers_blended
        """
        if p_student is None or alphas is None:
            if len(waveform) > 0:
                p_student, alphas = self.predict_student_tta(waveform)
            else:
                raise ValueError("Cần cung cấp waveform hoặc cặp (p_student, alphas).")

        if cached_t0_logits is not None and cached_tema_logits is not None:
            l_t0 = torch.from_numpy(cached_t0_logits).to(self.device).float()
            l_tema = torch.from_numpy(cached_tema_logits).to(self.device).float()
            p_t0 = F.softmax(l_t0 / self.t_teacher, dim=-1).cpu().numpy()
            p_tema = F.softmax(l_tema / self.t_teacher, dim=-1).cpu().numpy()
        else:
            assert self.teacher_fold0 is not None and self.teacher_ema is not None
            shifts = [0, 800, -800]
            l_t0_list, l_tema_list = [], []
            for s in shifts:
                l0, _ = tta_forward(self.teacher_fold0, waveform, self.device, shift_samples=s)
                lema, _ = tta_forward(self.teacher_ema, waveform, self.device, shift_samples=s)
                l_t0_list.append(l0)
                l_tema_list.append(lema)
            l_t0 = torch.stack(l_t0_list, dim=0).mean(dim=0)
            l_tema = torch.stack(l_tema_list, dim=0).mean(dim=0)
            p_t0 = F.softmax(l_t0 / self.t_teacher, dim=-1).cpu().numpy()
            p_tema = F.softmax(l_tema / self.t_teacher, dim=-1).cpu().numpy()

        a = alphas.reshape(-1, 1)
        w_t0 = (1.0 - a) * self.t0_low + a * self.t0_high
        w_tema = (1.0 - a) * self.tema_low + a * self.tema_high
        p_final = self.w_student * p_student + w_t0 * p_t0 + w_tema * p_tema

        return p_final, alphas


def load_pipeline(
    student_ckpt: Path | str,
    teacher_fold0_ckpt: Path | str | None = None,
    teacher_ema_ckpt: Path | str | None = None,
    beat_config_path: Path | str | None = None,
    device: torch.device | None = None,
) -> ChampionPipeline:
    """Nạp và khởi tạo toàn bộ pipeline hoàn chỉnh từ các file trọng số."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=False)
    ckpt_path = Path(student_ckpt)
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu")
        weights = state.get("model_state_dict", state)
        student.load_state_dict(weights, strict=False)
        print(f"Đã nạp thành công trọng số Student từ: {ckpt_path}")

    t0_model = None
    tema_model = None
    if teacher_fold0_ckpt and Path(teacher_fold0_ckpt).is_file():
        assert beat_config_path is not None
        t0_model = load_fold_teacher(teacher_fold0_ckpt, beat_config_path, dev)
    if teacher_ema_ckpt and Path(teacher_ema_ckpt).is_file():
        assert beat_config_path is not None
        tema_model = load_fold_teacher(teacher_ema_ckpt, beat_config_path, dev)

    return ChampionPipeline(student, t0_model, tema_model, device=dev)
