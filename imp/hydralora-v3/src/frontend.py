"""
Xử lý Âm thanh Frontend và Trích xuất Đặc trưng Âm học cho Kiến trúc Single-Backbone MoE.

Module này thực hiện các nhiệm vụ:
1. Chuẩn hóa Năng lượng RMS về mức tham chiếu sigma_0 = 0.07925 (~ -22.02 dBFS).
2. Trích xuất đặc trưng âm học vật lý (AcousticFeatureExtractor):
   - Biến đổi STFT tuyến tính với 257 dải tần số và 401 khung thời gian (16kHz, 4 giây = 64,000 mẫu).
   - 6 đặc trưng hình thái phổ và vật lý nội tại (f_1 đến f_6) phục vụ ước lượng SNR Zero-Oracle:
     * f_1: Mức năng lượng Log-RMS.
     * f_2: Hệ số đỉnh Crest factor (tỷ số giữa biên độ cực đại và RMS).
     * f_3: Hệ số nhọn năng lượng khung thời gian (Temporal Kurtosis - phát hiện xung đột biến rò rỉ tiếng nói từ FRCRN).
     * f_4: Độ phẳng phổ Spectral Flatness (Entropy Wiener: trung bình hình học / trung bình số học).
     * f_5: Tỷ số năng lượng dải trầm so với dải thoại (<300 Hz so với 300-3400 Hz).
     * f_6: Tỷ số năng lượng dải cao so với dải thoại (>3400 Hz so with 300-3400 Hz).
3. Tầng Frontend Log-Mel Spectrogram:
   - 64 dải lọc Mel, 398 khung thời gian, chuẩn hóa z-score độc lập cho từng mẫu.
4. Tăng cường dữ liệu SpecAugment:
   - Che dải tần số (Frequency Masking) và che dải thời gian (Time Masking).

Tài liệu tham khảo:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
    - SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", Interspeech 2019. arXiv:1904.08779.
    - Đơn vị đo âm học: Peeters, "A large set of audio features for sound description in the CUIDADO project", 2004.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RMS_EPS = 1e-8
TARGET_RMS = 0.07925  # Mức năng lượng âm học tham chiếu chuẩn hóa (~ -22.02 dBFS)


def rms_normalize(waveform: Tensor, target_rms: float = TARGET_RMS) -> Tensor:
    """
    Chuẩn hóa mức năng lượng hiệu dụng RMS của sóng âm về mức tham chiếu cố định target_rms (0.07925).

    Tại sao lại chọn target_rms = 0.07925?
    - Tương ứng với 20 * log10(0.07925) ~= -22.02 dBFS RMS, mức năng lượng trung vị thực nghiệm
      của tiếng nói sạch và nhiễu môi trường trong các bộ dữ liệu quy chuẩn AudioSet / FSD50K.
    - Tính bất biến (Invariance): Đảm bảo các đặc trưng âm học sau đó (Log-RMS, Crest factor,
      và tích phân năng lượng các dải tần) phản ánh cấu trúc phổ thật sự thay vì phụ thuộc vào
      âm lượng thu âm hoặc khoảng cách micro.
    - Tính ổn định (Stability): Chống hiện tượng bùng nổ hoặc tiêu biến gradient ở tầng tích chập patch Conv2d đầu tiên.

    Công thức:
        x_norm = x * [target_rms / (sqrt(mean(x^2)) + eps)]
    """
    rms = waveform.square().mean(dim=-1).sqrt()
    return waveform * (target_rms / (rms + RMS_EPS))[:, None]


def create_htk_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> Tensor:
    """
    Xây dựng ma trận dải lọc HTK Mel Filterbank chuẩn hóa, tương thích 100% với đặc tả của torchaudio và CED.
    """
    min_mel = 2595.0 * np.log10(1.0 + f_min / 700.0)
    max_mel = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(min_mel, max_mel, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        weights[i] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(weights)


class LogMel(nn.Module):
    """
    Tầng Frontend trích xuất phổ Log-Mel Spectrogram thuần PyTorch tương thích với CED:
    - Tần số lấy mẫu: 16 kHz
    - STFT: n_fft = 512, win_length = 400, hop_length = 160
    - Số dải Mel: 64, Dải tần số: 20 Hz - 8000 Hz
    - Số khung thời gian đầu ra: 398 frames
    - Chuẩn hóa z-score độc lập từng mẫu trên cả hai chiều không gian (thời gian, tần số).

    Tài liệu tham khảo:
        Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
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
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps
        self.pad = max(n_fft - win_length, 0)

        self.register_buffer("window", torch.hann_window(win_length))
        mel_fb = create_htk_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_fb", mel_fb)

    def logmel(self, waveform: Tensor) -> Tensor:
        """Tính toán phổ Log-Mel từ sóng âm đầu vào."""
        if self.pad:
            waveform = F.pad(waveform, (0, self.pad))
        # Thực hiện biến đổi STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = stft.abs().square()  # [B, n_fft // 2 + 1, Frames]
        # Nhân ma trận bộ lọc Mel: [n_mels, n_freqs] @ [B, n_freqs, Frames] -> [B, n_mels, Frames]
        mel = torch.matmul(self.mel_fb, power)
        return (mel + self.log_eps).log()

    def _offsets(self, frames: int, batch: int, device: torch.device) -> Tensor:
        """Sinh vị trí cắt khung thời gian (ngẫu nhiên khi huấn luyện, cắt giữa khi kiểm tra)."""
        limit = frames - self.num_frames
        if self.training and limit > 0:
            return torch.randint(0, limit + 1, (batch,), device=device)
        return torch.full((batch,), max(limit, 0) // 2, dtype=torch.long, device=device)

    def _crop(self, x: Tensor, offsets: Tensor) -> Tensor:
        """Cắt ma trận phổ về đúng độ dài khung chuẩn num_frames (398)."""
        frames = x.shape[-1]
        if frames <= self.num_frames:
            return x
        index = offsets[:, None] + torch.arange(self.num_frames, device=x.device)[None, :]
        return x.gather(2, index[:, None, :].expand(-1, x.shape[1], -1))

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Đầu vào: waveform [B, 64000]
        Đầu ra: Log-Mel spectrogram [B, 1, 64, 398] đã chuẩn hóa z-score.
        """
        x = self.logmel(waveform)
        # Chuẩn hóa z-score cho từng mẫu qua cả chiều tần số và thời gian
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + self.norm_eps)
        return self._crop(x, self._offsets(x.shape[-1], x.shape[0], x.device)).unsqueeze(1)


class SpecAugment(nn.Module):
    """
    Kỹ thuật tăng cường dữ liệu SpecAugment che dải tần số và thời gian.

    Tài liệu tham khảo:
        Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", Interspeech 2019. arXiv:1904.08779.
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
        """Che về 0 một dải ngẫu nhiên trên chiều dim chỉ định."""
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
        """Áp dụng che dải (chỉ kích hoạt khi đang ở chế độ huấn luyện training=True)."""
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x
        x = x.clone()
        for sample in range(x.shape[0]):
            for _ in range(self.freq_masks):
                self._zero_band(x, sample, dim=2, width=self.freq_width)
            for _ in range(self.time_masks):
                self._zero_band(x, sample, dim=3, width=self.time_width)
        return x


class AcousticFeatureExtractor(nn.Module):
    """
    Trích xuất 6 đặc trưng vật lý, phổ và hình thái thống kê trực tiếp từ sóng âm:
    Đầu vào: Sóng âm đã chuẩn hóa n_hat độ dài 4.0 giây (64,000 mẫu tại 16 kHz).
    Biến đổi STFT:
        - n_fft = 512 -> 257 dải tần số tuyến tính (0 đến 8000 Hz, delta_f = 31.25 Hz).
        - hop_length = 160 -> bước nhảy 10 ms.
        - win_length = 512 với center=True -> 401 khung thời gian (1 + 64000 / 160).
        - Sinh ra phổ công suất tuyến tính P(f, t) kích thước [B, 257, 401].

    6 Đặc trưng Âm học Vật lý [f_1, f_2, f_3, f_4, f_5, f_6]:
        1. f_1 (Log-RMS): Mức năng lượng tổng thể của tín hiệu sau chuẩn hóa.
        2. f_2 (Crest Factor): Tỷ số giữa biên độ cực đại và năng lượng hiệu dụng. Nhận diện xung kích đột biến.
        3. f_3 (Temporal Kurtosis): Hệ số nhọn bậc 4 của năng lượng khung theo thời gian.
           Nhiễu môi trường nền có kurtosis xấp xỉ 3. Khi bộ khử nhiễu FRCRN làm rò rỉ âm tiết tiếng nói
           ngắt quãng ở SNR cao, Kurtosis sẽ tăng đột biến (>> 3).
        4. f_4 (Spectral Flatness): Tỷ số trung bình hình học trên trung bình số học (Entropy Wiener).
           Xấp xỉ 1 đối với nhiễu phẳng (nhiễu trắng/hồng); rất nhỏ (<< 1) khi có cấu trúc họa âm tiếng người.
        5. f_5 (Low-band Energy Ratio): Tỷ số log năng lượng dải trầm (<300 Hz, bins 0..9)
           so với dải thoại người (300-3400 Hz, bins 10..108).
        6. f_6 (High-band Energy Ratio): Tỷ số log năng lượng dải cao (>3400 Hz, bins 109..256)
           so với dải thoại người (300-3400 Hz, bins 10..108).

    Ý nghĩa:
        Cung cấp thước đo định lượng biến dạng vật lý và tàn dư rò rỉ tiếng nói từ FRCRN
        mà không đòi hỏi nhãn SNR Oracle hay tín hiệu tiếng nói gốc!
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 160) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def compute_stft_power(self, waveform: Tensor) -> Tensor:
        """
        Tính toán ma trận mật độ phổ công suất tuyến tính P(f, t) kích thước [B, 257, 401]
        tương ứng với đầu vào 64,000 mẫu âm thanh.
        """
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )  # [B, 257, 401]
        power = stft.abs().square()  # [B, 257, 401]
        return power

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Đầu vào: waveform [B, 64000]
        Đầu ra: features [B, 6] tương ứng [f_1, f_2, f_3, f_4, f_5, f_6].
        """
        B, T_len = waveform.shape
        eps = 1e-8

        # 1. f_1: Log-RMS
        rms = waveform.square().mean(dim=-1).clamp(min=eps).sqrt()
        log_rms = torch.log(rms + eps).unsqueeze(-1)  # [B, 1]

        # 2. f_2: Hệ số đỉnh Crest factor (peak / rms)
        peak = waveform.abs().max(dim=-1).values
        crest_factor = (peak / (rms + eps)).unsqueeze(-1)  # [B, 1]

        # Tính phổ công suất tuyến tính P(f, t) kích thước [B, 257, 401]
        power = self.compute_stft_power(waveform) + eps

        # 3. f_3: Hệ số nhọn năng lượng khung thời gian (Temporal Kurtosis)
        frame_energy = power.sum(dim=1)  # [B, 401]
        mean_fe = frame_energy.mean(dim=-1, keepdim=True)
        std_fe = frame_energy.std(dim=-1, keepdim=True) + eps
        kurtosis = (((frame_energy - mean_fe) / std_fe) ** 4).mean(dim=-1, keepdim=True)  # [B, 1]

        # 4. f_4: Độ phẳng phổ (Spectral Flatness / Wiener entropy)
        log_power = torch.log(power)
        geom_mean = torch.exp(log_power.mean(dim=1))  # [B, 401]
        arith_mean = power.mean(dim=1)                # [B, 401]
        flatness = (geom_mean / (arith_mean + eps)).mean(dim=-1, keepdim=True)  # [B, 1]

        # 5 & 6. Tỷ số năng lượng dải tần số (Sub-band energy ratios):
        # Độ phân giải dải tần: 16000 / 512 = 31.25 Hz/bin
        # Dải trầm (<300 Hz): bins 0..9 (0 đến 312.5 Hz)
        # Dải thoại tiếng nói (300-3400 Hz): bins 10..108 (312.5 đến 3406.25 Hz)
        # Dải cao (>3400 Hz): bins 109..256 (3406.25 đến 8000 Hz)
        band_low = power[:, :10, :].sum(dim=(1, 2)) + eps
        band_speech = power[:, 10:109, :].sum(dim=(1, 2)) + eps
        band_high = power[:, 109:, :].sum(dim=(1, 2)) + eps

        ratio_low_speech = torch.log(band_low / band_speech).unsqueeze(-1)    # [B, 1]
        ratio_high_speech = torch.log(band_high / band_speech).unsqueeze(-1)  # [B, 1]

        features = torch.cat(
            [log_rms, crest_factor, kurtosis, flatness, ratio_low_speech, ratio_high_speech],
            dim=-1,
        )  # [B, 6]
        return features
