from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_snr_db(mixture: torch.Tensor, noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Ước lượng tỷ số tín hiệu trên nhiễu (Signal-to-Noise Ratio - SNR) theo đơn vị dB.
    
    Nguyên lý:
        - Tín hiệu sạch ước lượng: clean = mixture - noise
        - Công suất tín hiệu sạch: P_clean = Mean(clean^2)
        - Công suất tín hiệu nhiễu: P_noise = Mean(noise^2)
        - SNR (dB) = 10 * log10(P_clean / P_noise)

    Tham số:
        mixture: Tensor âm thanh hỗn hợp [B, T] hoặc [T]
        noise: Tensor nhiễu được tách/ước lượng [B, T] hoặc [T]
        eps: Hệ số làm mịn tránh chia cho 0 hoặc log(0)

    Trả về:
        Tensor SNR theo dB cho từng mẫu trong batch [B]
    """
    if mixture.ndim == 1:
        mixture = mixture.unsqueeze(0)
    if noise.ndim == 1:
        noise = noise.unsqueeze(0)

    # Khôi phục tín hiệu sạch giả định: s = x_mix - n
    clean = mixture - noise
    # Tính công suất trung bình của tín hiệu và nhiễu
    power_clean = torch.mean(clean**2, dim=-1) + eps
    power_noise = torch.mean(noise**2, dim=-1) + eps
    return 10.0 * torch.log10(power_clean / power_noise)


class EnergySNRRouter:
    """
    Bộ định tuyến động (Hard Gating Router) dựa trên ngưỡng năng lượng SNR.
    
    Quy tắc định tuyến:
        - Nếu SNR >= threshold_db (mặc định 18.0 dB): Tuyến High-SNR (môi trường ít nhiễu / âm thanh rõ).
          Ưu tiên Teacher EMA và Student High-Expert.
        - Nếu SNR < threshold_db: Tuyến Low-SNR (môi trường nhiễu mạnh, SNR từ -5 dB đến 15 dB).
          Ưu tiên Teacher Fold0 và Student Low-Expert.
    """
    def __init__(self, threshold_db: float = 18.0, eps: float = 1e-8) -> None:
        self.threshold_db = float(threshold_db)
        self.eps = float(eps)

    def route(self, snrs_db: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Phân loại mảng SNR thành mặt nạ nhị phân (True: High SNR, False: Low SNR).
        """
        if isinstance(snrs_db, torch.Tensor):
            snrs = snrs_db.detach().cpu().numpy()
        else:
            snrs = np.asarray(snrs_db)
        return snrs >= self.threshold_db

    def estimate_and_route(
        self, mixture: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vừa ước lượng giá trị SNR từ waveform, vừa trả về mặt nạ phân nhánh (is_high).
        """
        snrs = estimate_snr_db(mixture, noise, eps=self.eps)
        is_high = snrs >= self.threshold_db
        return snrs, is_high


class NeuralRouter(nn.Module):
    """
    Bộ định tuyến học sâu (Soft Gating Router) dùng mạng nơ-ron MLP nhẹ.
    
    Thay vì quyết định rẽ nhánh nhị phân (0 hoặc 1), NeuralRouter ước tính trọng số mềm
    alpha in [0, 1] để nội suy liên tục giữa Low-Expert và High-Expert.
    
    Kiến trúc:
        - Đầu vào: 4 đặc trưng năng lượng (RMS mix, RMS noise, tỷ lệ nhiễu, SNR chuẩn hóa).
        - MLP: Linear(4, 16) -> GELU -> Linear(16, 1) -> Sigmoid -> alpha
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def extract_features(
        self, mixture: torch.Tensor, noise: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Trích xuất 4 đặc trưng âm học cốt lõi phục vụ định tuyến:
            1. RMS của âm thanh gốc (Mixture)
            2. RMS của thành phần nhiễu (Noise)
            3. Tỉ lệ năng lượng nhiễu trên hỗn hợp (Noise-to-Mix Ratio)
            4. SNR ước lượng thu nhỏ tỷ lệ (SNR / 20 dB)
        """
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if noise.ndim == 1:
            noise = noise.unsqueeze(0)

        # 1 & 2. Cường độ âm lượng RMS
        rms_mix = torch.sqrt(torch.mean(mixture**2, dim=-1, keepdim=True) + eps)
        rms_noise = torch.sqrt(torch.mean(noise**2, dim=-1, keepdim=True) + eps)
        # 3. Tỷ lệ tương đối
        energy_ratio = rms_noise / (rms_mix + eps)

        # 4. SNR ước lượng
        clean = mixture - noise
        power_clean = torch.mean(clean**2, dim=-1, keepdim=True) + eps
        power_noise = torch.mean(noise**2, dim=-1, keepdim=True) + eps
        snr_est = 10.0 * torch.log10(power_clean / power_noise)

        # Nối thành vector 4 chiều: [B, 4]
        return torch.cat([rms_mix, rms_noise, energy_ratio, snr_est / 20.0], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Dự đoán xác suất định tuyến mềm alpha [B, 1]"""
        return self.net(features)

    def route_soft(self, mixture: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Tiện ích trích xuất đặc trưng và tính toán routing weight trực tiếp từ waveform [B]"""
        feats = self.extract_features(mixture, noise)
        return self.forward(feats).squeeze(-1)
