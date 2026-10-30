from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.metrics import snr_scores
from src.model.beat import load_fold_teacher
from src.model.student import CEDStudent, build_student
from src.router import EnergySNRRouter
from src.utils.files import rms_normalize


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """
    Hàm Softmax kèm hệ số nhiệt độ (Temperature Scaling).
    
    Công thức:
        P_i = exp(z_i / T) / sum(exp(z_j / T))
        
    Mục đích:
        - T = 1.0: Softmax thông thường.
        - T > 1.0: Làm mềm phân phối xác suất (giảm bớt sự tự tin thái quá / over-confidence).
        - Trừ max(x) trước khi exp để ổn định số học (tránh tràn số overflow).
    """
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return e / np.sum(e, axis=-1, keepdims=True)


def tta_forward(
    model: torch.nn.Module,
    wave: torch.Tensor,
    device: torch.device,
    shift_samples: int = 0,
    rms: float = 0.07925,
) -> torch.Tensor:
    """
    Thực hiện 1 lượt dự đoán (forward pass) với độ dịch chuyển thời gian (Temporal Shift).
    
    Quy trình:
        1. Dịch chuyển tín hiệu wave theo `shift_samples` (dùng reflect padding để giữ nguyên độ dài).
        2. Chuẩn hóa biên độ năng lượng RMS về 0.07925 (chuẩn của tập dữ liệu huấn luyện).
        3. Dự đoán qua mô hình với cơ chế Automatic Mixed Precision (bfloat16) nếu chạy trên CUDA.
    
    Tham số:
        model: Mô hình nơ-ron (Student hoặc Teacher)
        wave: Tensor dạng sóng âm thanh [B, 64000]
        shift_samples: Số mẫu dịch chuyển (+800 mẫu ứng với +50ms ở 16kHz, -800 mẫu là -50ms)
        rms: Giá trị RMS mục tiêu
    """
    if shift_samples != 0:
        if shift_samples > 0:
            # Dịch sang trái và pad đuôi bằng mode reflect
            w = F.pad(wave[:, shift_samples:], (0, shift_samples), mode="reflect")
        else:
            # Dịch sang phải và pad đầu bằng mode reflect
            s = abs(shift_samples)
            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
    else:
        w = wave

    # Chuẩn hóa mức năng lượng đồng nhất
    noise = rms_normalize(w, rms)
    enabled = device.type == "cuda"
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
        logits = model(noise).logits
    return logits.float()


def run_tta_predict(
    model: torch.nn.Module,
    wave: torch.Tensor,
    device: torch.device,
    shift_delta: int = 800,
) -> torch.Tensor:
    """
    Chiến lược Test-Time Augmentation (3-Way Temporal TTA).
    
    Tổng hợp logits từ 3 góc nhìn thời gian:
        - Tín hiệu gốc: l0 (shift = 0) với trọng số 1.0
        - Tín hiệu dịch tới: lp (shift = +50ms / +800 mẫu) với trọng số 0.5
        - Tín hiệu dịch lùi: ln (shift = -50ms / -800 mẫu) với trọng số 0.5
        
    Công thức trung bình gia quyền:
        z = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
    
    Lợi ích:
        Giảm thiểu hiện tượng boundary artifact và bất biến với sai số cắt khung tín hiệu.
    """
    l0 = tta_forward(model, wave, device, shift_samples=0)
    lp = tta_forward(model, wave, device, shift_samples=shift_delta)
    ln = tta_forward(model, wave, device, shift_samples=-shift_delta)
    return (l0 + 0.5 * lp + 0.5 * ln) / 2.0


class ChampionPipeline:
    """
    Hệ thống Champion Pipeline: Dual-Expert & Dual-Teacher Mixture of Experts (MoE).
    
    Bao gồm 4 động cơ nơ-ron hoạt động song song kết hợp định tuyến động theo mức năng lượng SNR:
        1. S1: CED-Tiny Low-Student (5.42M params) - Chuyên gia khử nhiễu mạnh (SNR < 18 dB).
        2. S2: CED-Tiny High-Student (5.42M params) - Chuyên gia phân tách tinh xảo (SNR >= 18 dB).
        3. T1: BEATs Teacher 12L Fold0 (90.34M params) - Thầy tổng quát huấn luyện trên Clean + Noise.
        4. T2: BEATs Teacher 12L EMA (90.34M params) - Thầy lọc nhiễu FRCRN kết hợp trung bình trọng số di động.
        
    Cơ chế phối hợp:
        - Tuyến Low (SNR < 18 dB): P = 0.30 * S_low + 0.45 * T_fold0 + 0.25 * T_ema
        - Tuyến High (SNR >= 18 dB): P = 0.35 * S_high + 0.40 * T_ema + 0.25 * T_fold0
    """
    def __init__(
        self,
        low_student: CEDStudent,
        high_student: CEDStudent,
        teacher_fold0: torch.nn.Module,
        teacher_ema: torch.nn.Module,
        device: torch.device,
        snr_threshold_db: float = 18.0,
        temp_student: float = 1.40,
        temp_teacher: float = 1.20,
        low_weights: tuple[float, float, float] = (0.30, 0.45, 0.25),
        high_weights: tuple[float, float, float] = (0.35, 0.40, 0.25),
    ) -> None:
        self.low_student = low_student.to(device).eval()
        self.high_student = high_student.to(device).eval()
        self.teacher_fold0 = teacher_fold0.to(device).eval()
        self.teacher_ema = teacher_ema.to(device).eval()
        self.device = device
        self.snr_threshold = snr_threshold_db
        self.temp_s = temp_student
        self.temp_t = temp_teacher
        self.w_low = low_weights
        self.w_high = high_weights
        self.router = EnergySNRRouter(threshold_db=snr_threshold_db)

    @torch.no_grad()
    def predict_batch(
        self,
        wave: torch.Tensor,
        snrs_db: np.ndarray | torch.Tensor | None = None,
        mixture: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Dự đoán phân phối xác suất cho một batch âm thanh.
        
        Quy trình:
            1. Xác định mức SNR: trực tiếp từ tham số snrs_db hoặc ước lượng tự động từ (mixture, wave).
            2. Chạy 3-Way TTA cho 2 Teachers (Fold0 và EMA) trên toàn bộ batch.
            3. Hiệu chuẩn nhiệt độ không đối xứng (Asymmetric Temperature Calibration):
                 - p_teacher = Softmax(z_t / 1.20)
                 - p_student = Softmax(z_s / 1.40)
            4. Định tuyến theo phân nhánh:
                 - Nhánh Low: chỉ chạy Low-Student trên các mẫu có SNR < 18 dB.
                 - Nhánh High: chỉ chạy High-Student trên các mẫu có SNR >= 18 dB.
            5. Tổng hợp trọng số xác suất (Probability Blending) và trả về mảng xác suất [B, 36].
        """
        wave = wave.to(self.device, non_blocking=True)

        # 1. Xác định cờ phân nhánh High/Low SNR cho từng clip
        if snrs_db is None:
            if mixture is not None:
                mixture = mixture.to(self.device, non_blocking=True)
                snrs_t, is_high_t = self.router.estimate_and_route(mixture, wave)
                is_high_snr = is_high_t.cpu().numpy()
            else:
                is_high_snr = np.zeros(wave.shape[0], dtype=bool)
        elif isinstance(snrs_db, torch.Tensor):
            is_high_snr = (snrs_db >= self.snr_threshold).cpu().numpy()
        else:
            is_high_snr = np.asarray(snrs_db) >= self.snr_threshold

        # 2. Suy luận qua 2 giáo viên nền tảng (BEATs Fold0 và BEATs EMA)
        t0_logits = run_tta_predict(self.teacher_fold0, wave, self.device)
        tema_logits = run_tta_predict(self.teacher_ema, wave, self.device)

        # 3. Hiệu chuẩn nhiệt độ Teacher (T=1.20)
        p_t0 = softmax(t0_logits.cpu().numpy(), temp=self.temp_t)
        p_tema = softmax(tema_logits.cpu().numpy(), temp=self.temp_t)

        final_probs = np.zeros_like(p_t0)

        # 4a. Định tuyến Tuyến Low (SNR < 18 dB)
        if not np.all(is_high_snr):
            low_mask = ~is_high_snr
            w_low = wave[low_mask]
            s_low_logits = run_tta_predict(self.low_student, w_low, self.device)
            p_s_low = softmax(s_low_logits.cpu().numpy(), temp=self.temp_s)

            final_probs[low_mask] = (
                self.w_low[0] * p_s_low
                + self.w_low[1] * p_t0[low_mask]
                + self.w_low[2] * p_tema[low_mask]
            )

        # 4b. Định tuyến Tuyến High (SNR >= 18 dB)
        if np.any(is_high_snr):
            high_mask = is_high_snr
            w_high = wave[high_mask]
            s_high_logits = run_tta_predict(self.high_student, w_high, self.device)
            p_s_high = softmax(s_high_logits.cpu().numpy(), temp=self.temp_s)

            final_probs[high_mask] = (
                self.w_high[0] * p_s_high
                + self.w_high[1] * p_tema[high_mask]
                + self.w_high[2] * p_t0[high_mask]
            )

        return final_probs

    def predict_proba(
        self,
        wave: torch.Tensor,
        snrs_db: np.ndarray | torch.Tensor | None = None,
        mixture: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Trả về ma trận phân phối xác suất [B, Num_Classes]"""
        return self.predict_batch(wave, snrs_db=snrs_db, mixture=mixture)

    def predict(
        self,
        wave: torch.Tensor,
        snrs_db: np.ndarray | torch.Tensor | None = None,
        mixture: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Dự đoán nhãn lớp đơn lẻ có xác suất cao nhất: argmax(probs) -> [B]"""
        probs = self.predict_proba(wave, snrs_db=snrs_db, mixture=mixture)
        return np.argmax(probs, axis=-1)


MoEPipeline = ChampionPipeline


def load_pipeline(
    config_path: str | Path,
    low_ckpt: str | Path,
    high_ckpt: str | Path,
    t_fold0_ckpt: str | Path,
    t_ema_ckpt: str | Path,
    device: torch.device,
    beat_config_path: str | Path | None = None,
) -> ChampionPipeline:
    """
    Hàm khởi tạo và nạp toàn bộ trọng số cho ChampionPipeline từ file cấu hình YAML và Checkpoints.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if beat_config_path is None:
        beat_config_path = config_path.parent / "beat.yaml"
    else:
        beat_config_path = Path(beat_config_path)

    # Khởi tạo và nạp trọng số S1: Low-Student
    low_student = build_student(cfg, root=config_path.parent)
    low_student.load_pretrained(Path(low_ckpt))

    # Khởi tạo và nạp trọng số S2: High-Student
    high_student = build_student(cfg, root=config_path.parent)
    high_data = torch.load(high_ckpt, map_location="cpu", weights_only=False)
    high_student.load_state_dict(high_data["model"])

    # Khởi tạo và nạp trọng số 2 mô hình Teacher BEATs 12L
    t_fold0, _ = load_fold_teacher(Path(t_fold0_ckpt), beat_config_path, device)
    t_ema, _ = load_fold_teacher(Path(t_ema_ckpt), beat_config_path, device)

    pipe_cfg = cfg.get("pipeline", {})
    snr_thresh = pipe_cfg.get("snr_threshold_db", 18.0)
    temp_s = pipe_cfg.get("temp_student", 1.40)
    temp_t = pipe_cfg.get("temp_teacher", 1.20)

    low_w = pipe_cfg.get("low_expert", {})
    low_weights = (
        float(low_w.get("student_weight", 0.30)),
        float(low_w.get("teacher_fold0_weight", 0.45)),
        float(low_w.get("teacher_ema_weight", 0.25)),
    )

    high_w = pipe_cfg.get("high_expert", {})
    high_weights = (
        float(high_w.get("student_weight", 0.35)),
        float(high_w.get("teacher_ema_weight", 0.40)),
        float(high_w.get("teacher_fold0_weight", 0.25)),
    )

    return ChampionPipeline(
        low_student,
        high_student,
        t_fold0,
        t_ema,
        device,
        snr_threshold_db=snr_thresh,
        temp_student=temp_s,
        temp_teacher=temp_t,
        low_weights=low_weights,
        high_weights=high_weights,
    )


def evaluate_pipeline(
    pipeline: ChampionPipeline,
    data_root: str | Path,
    cache_dir: str | Path,
    split: str = "validation",
    batch_size: int = 64,
) -> dict[str, Any]:
    """
    Quy trình đánh giá toàn diện Champion Pipeline trên tập dữ liệu kiểm định.
    
    Các bước:
        1. Kết nối Mixture36Dataset với cache npy mmap (tốc độ cao).
        2. Duyệt qua từng batch và thực hiện dự đoán phân phối xác suất.
        3. Tính toán toàn bộ 9 metrics và phân rã hiệu năng chi tiết theo từng mốc SNR (-5 dB đến 20 dB).
    """
    base = Mixture36Dataset(root=Path(data_root), split=split, duration=4.0)
    cache_file = Path(cache_dir) / f"{split}.npy"
    done_file = Path(cache_dir) / f"{split}.done"
    cache_dataset = NoiseCacheDataset(base, cache_file, done_file)
    loader = build_dataloader(cache_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_snrs: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"Evaluating MoE Champion Pipeline ({split})"):
        wave = batch["waveform"]
        snrs = batch["snr_db"].numpy()
        targets = batch["target"].argmax(dim=-1).numpy()

        probs = pipeline.predict_batch(wave, snrs)
        all_probs.append(probs)
        all_labels.append(targets)
        all_snrs.append(snrs)

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)

    return snr_scores(probs_arr, labels_arr, snrs_arr, 36)
