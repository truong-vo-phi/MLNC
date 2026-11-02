from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Final
from dataclasses import dataclass

RMS_EPS: Final[float] = 1e-8
TARGET_RMS: Final[float] = 0.07925 # Mức năng lượng chuẩn tham chiếu (~ -22.02 dBFS)

# 1. Hàm chuẩn hoá năng lượng RMS
def rms_normalize(waveform: Tensor, target_rms: float = TARGET_RMS) -> Tensor:
    """
    Đưa mọi tín hiệu sóng âm về cùng mức năng lượng hiệu dụng target_rms = 0.07925.
    Công thức: x_norm = x * (target_rms / (sqrt(mean(x^2)) + eps))
    """
    # Tính RMS dọc theo trục thời gian (dim=-1): [B, 64000] -> [B]
    rms = waveform.square().mean(dim=-1).sqrt()
    # Nhân hệ số tỷ lệ cho từng mẫu trong batch ([:, None] mở rộng chiều về [B, 1])
    return waveform * (target_rms / (rms + RMS_EPS))[:, None]

# 2. Hàm tạo bộ lọc HTK Mel Filterbank thuần Pytorch (Hidden Markov Model Toolkit)
def create_htk_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> Tensor:
    """
    Khởi tạo ma trận bộ lọc tam giác Mel theo chuẩn HTK: kích thước [64, 257].
    Viết thuần bằng NumPy/PyTorch để không phụ thuộc vào C++ extension của torchaudio.
    """
    # Chuyển tần số giới hạn Hz sang thang Mel: m = 2595 * log10(1 + f / 700)
    min_mel = 2595.0 * np.log10(1.0 + f_min / 700.0)
    max_mel = 2595.0 * np.log10(1.0 + f_max / 700.0)
    # Chia đều n_mels + 2 điểm trên thang Mel
    mel_pts = np.linspace(min_mel, max_mel, n_mels + 2)
    # Đổi ngược từ Mel về lại Hz: f = 700 * (10^(m / 2595) - 1)
    f_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    # Tần số tuyến tính của 257 bins FFT (từ 0 Hz đến 8000 Hz)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        # Sườn lên và sườn xuống của bộ lọc tam giác thứ i
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        weights[i] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(weights)

# 3. Nhánh phổ Log-Mel & Specaugment cho backbne ViT
class LogMel(nn.Module):
    """
    Biến đổi sóng âm [B, 64000] thành phổ Log-Mel chuẩn hóa z-score [B, 1, 64, 398].
    Khớp 100% với cấu hình tiền huấn luyện của CED-Tiny trên AudioSet.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        num_frames: int = 398,
        log_eps: float = 1e-6,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps
        # Đệm thêm (512 - 400) = 112 mẫu ở cuối để đủ cửa sổ FFT khi center=False
        self.pad = max(n_fft - win_length, 0)
        # Đăng ký buffer để window và mel_fb tự động chuyển sang GPU cùng model
        self.register_buffer("window", torch.hann_window(win_length))
        mel_fb = create_htk_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_fb", mel_fb)
    def logmel(self, waveform: Tensor) -> Tensor:
        if self.pad:
            waveform = F.pad(waveform, (0, self.pad))
        # STFT với center=False: tạo ra (64112 - 512) // 160 + 1 = 398 khung thời gian
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = stft.abs().square()  # [B, 257, 398]
        # Nhân ma trận: [64, 257] @ [B, 257, 398] -> [B, 64, 398]
        mel = torch.matmul(self.mel_fb, power)
        return (mel + self.log_eps).log()
    def _offsets(self, frames: int, batch: int, device: torch.device) -> Tensor:
        limit = frames - self.num_frames
        if self.training and limit > 0:
            return torch.randint(0, limit + 1, (batch,), device=device)
        return torch.full((batch,), max(limit, 0) // 2, dtype=torch.long, device=device)
    def _crop(self, x: Tensor, offsets: Tensor) -> Tensor:
        frames = x.shape[-1]
        if frames <= self.num_frames:
            return x
        index = offsets[:, None] + torch.arange(self.num_frames, device=x.device)[None, :]
        return x.gather(2, index[:, None, :].expand(-1, x.shape[1], -1))
    def forward(self, waveform: Tensor) -> Tensor:
        x = self.logmel(waveform)  # [B, 64, 398]
        # Chuẩn hóa Z-score độc lập cho từng mẫu âm thanh trên cả 2 chiều (tần số, thời gian)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + self.norm_eps)
        # Cắt đúng 398 frames và thêm chiều kênh ảnh (dim=1) -> [B, 1, 64, 398]
        return self._crop(x, self._offsets(x.shape[-1], x.shape[0], x.device)).unsqueeze(1)

class SpecAugment(nn.Module):
    """
    Tăng cường dữ liệu bằng cách che ngẫu nhiên các dải tần số và thời gian (chỉ bật khi training).
    """
    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 8,
        time_masks: int = 2,
        time_width: int = 24,
    ) -> None:
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width
    @staticmethod
    def _zero_band(x: Tensor, sample: int, dim: int, width: int) -> None:
        size = x.shape[dim]
        w = int(torch.randint(0, width + 1, (1,)).item())
        if w == 0 or w > size:
            return
        start = int(torch.randint(0, size - w + 1, (1,)).item())
        if dim == 2:
            x[sample, :, start : start + w, :] = 0
        else:
            x[sample, :, :, start : start + w] = 0
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x
        x = x.clone()
        for sample in range(x.shape[0]):
            for _ in range(self.freq_masks):
                self._zero_band(x, sample, dim=2, width=self.freq_width)
            for _ in range(self.time_masks):
                self._zero_band(x, sample, dim=3, width=self.time_width)
        return x

# 4. Bộ trích xuất 6 đặc trưng âm học vật lý (Không cần nhãn SNR)
class AcousticFeatureExtractor(nn.Module):
    """
    Trích xuất 6 đặc trưng vật lý f_spec ∈ R^[B, 6] từ sóng âm đã chuẩn hóa:
      - f1: Log-RMS (năng lượng tổng thể)
      - f2: Crest factor (tỷ số đỉnh / hiệu dụng)
      - f3: Kurtosis (độ nhọn phân phối năng lượng khung -> bắt xung rò rỉ tiếng nói)
      - f4: Spectral Flatness (độ phẳng phổ Wiener -> phân biệt nhiễu trắng và âm điều hòa)
      - f5: Tỷ số năng lượng dải trầm (<300 Hz) so với dải thoại (300-3400 Hz)
      - f6: Tỷ số năng lượng dải cao (>3400 Hz) so với dải thoại (300-3400 Hz)
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 160) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
    def compute_stft_power(self, waveform: Tensor) -> Tensor:
        """
        Tính phổ công suất tuyến tính P(f, t) với center=True:
        Đầu ra có kích thước chính xác [B, 257, 401] cho đoạn âm thanh 64,000 mẫu.
        """
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return stft.abs().square()  # [B, 257, 401]
    def forward(self, waveform: Tensor) -> Tensor:
        eps = 1e-8
        # f1: Log-RMS
        rms = waveform.square().mean(dim=-1).clamp(min=eps).sqrt()
        log_rms = torch.log(rms + eps).unsqueeze(-1)  # [B, 1]
        # f2: Crest factor = max(|x|) / RMS
        peak = waveform.abs().max(dim=-1).values
        crest_factor = (peak / (rms + eps)).unsqueeze(-1)  # [B, 1]
        # Phổ công suất P(f, t): [B, 257, 401]
        power = self.compute_stft_power(waveform) + eps
        # f3: Kurtosis bậc 4 của năng lượng theo từng khung thời gian
        frame_energy = power.sum(dim=1)  # Tổng năng lượng theo tần số -> [B, 401]
        mean_fe = frame_energy.mean(dim=-1, keepdim=True)
        std_fe = frame_energy.std(dim=-1, keepdim=True) + eps
        kurtosis = (((frame_energy - mean_fe) / std_fe) ** 4).mean(dim=-1, keepdim=True)  # [B, 1]
        # f4: Spectral Flatness = Trung bình hình học / Trung bình số học theo trục tần số
        log_power = torch.log(power)
        geom_mean = torch.exp(log_power.mean(dim=1))  # [B, 401]
        arith_mean = power.mean(dim=1)                # [B, 401]
        flatness = (geom_mean / (arith_mean + eps)).mean(dim=-1, keepdim=True)  # [B, 1]
        # f5 & f6: Tỷ số năng lượng dải tần (Độ phân giải 1 bin = 16000 / 512 = 31.25 Hz)
        # - Bins 0..9:     0 Hz -> 312.5 Hz (Dải trầm < 300 Hz)
        # - Bins 10..108:  312.5 Hz -> 3406.25 Hz (Dải thoại 300 - 3400 Hz)
        # - Bins 109..256: 3406.25 Hz -> 8000 Hz (Dải cao > 3400 Hz)
        band_low = power[:, :10, :].sum(dim=(1, 2)) + eps
        band_speech = power[:, 10:109, :].sum(dim=(1, 2)) + eps
        band_high = power[:, 109:, :].sum(dim=(1, 2)) + eps
        ratio_low_speech = torch.log(band_low / band_speech).unsqueeze(-1)    # [B, 1]
        ratio_high_speech = torch.log(band_high / band_speech).unsqueeze(-1)  # [B, 1]
        # Ghép 6 đặc trưng lại thành [B, 6]
        return torch.cat(
            [log_rms, crest_factor, kurtosis, flatness, ratio_low_speech, ratio_high_speech],
            dim=-1,
        )