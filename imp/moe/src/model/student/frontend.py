from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn
from torchaudio import transforms as T

from src.core import BaseConfig


@dataclass(frozen=True)
class LogMelConfig(BaseConfig):
    section: ClassVar[str] = "frontend"
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float = 8000.0
    num_frames: int = 398
    log_eps: float = 1.0e-6
    norm_eps: float = 1.0e-5


@dataclass(frozen=True)
class SpecAugmentConfig(BaseConfig):
    section: ClassVar[str] = "specaugment"
    freq_masks: int = 2
    freq_width: int = 16
    time_masks: int = 2
    time_width: int = 24


class LogMel(nn.Module):
    """
    Module trích xuất phổ Log-Mel Spectrogram từ dạng sóng âm thanh (Waveform).
    
    Thông số kỹ thuật:
        - Tần số lấy mẫu: 16,000 Hz
        - Cửa sổ STFT: win_length = 400 mẫu (25 ms)
        - Độ dịch bước: hop_length = 160 mẫu (10 ms)
        - Số dải lọc Mel: n_mels = 64 bins (tần số 20 Hz - 8000 Hz)
        - Số khung thời gian: 398 frames (tương đương 4.0 giây)
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
        self.num_frames = num_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps
        self.pad = max(n_fft - win_length, 0)
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=False,
        )

    def logmel(self, waveform: Tensor) -> Tensor:
        """Tính toán log-mel spectrogram với padding nếu n_fft > win_length"""
        if self.pad:
            waveform = nn.functional.pad(waveform, (0, self.pad))
        return (self.melspec(waveform) + self.log_eps).log()

    def _offsets(self, frames: int, batch: int, device: torch.device) -> Tensor:
        """
        Tính điểm bắt đầu cắt khung (offset):
            - Trong quá trình Train: Cắt ngẫu nhiên (Random Crop) để tăng tính đa dạng.
            - Trong quá trình Eval/Test: Cắt ở chính giữa (Center Crop) để đảm bảo tính tất định.
        """
        limit = frames - self.num_frames
        if self.training and limit > 0:
            return torch.randint(0, limit + 1, (batch,), device=device)
        return torch.full((batch,), max(limit, 0) // 2, dtype=torch.long, device=device)

    def _crop(self, x: Tensor, offsets: Tensor) -> Tensor:
        """Cắt ma trận phổ theo trục thời gian đúng kích thước num_frames (398 frames)"""
        frames = x.shape[-1]
        if frames <= self.num_frames:
            return x
        index = offsets[:, None] + torch.arange(self.num_frames, device=x.device)[None, :]
        return x.gather(2, index[:, None, :].expand(-1, x.shape[1], -1))

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Đầu vào: Waveform [B, Num_Samples] (ví dụ: [B, 64000])
        Đầu ra: Log-Mel chuẩn hóa [B, 1, Num_Mel_Bins, Num_Frames] (ví dụ: [B, 1, 64, 398])
        """
        x = self.logmel(waveform)
        # Chuẩn hóa từng mẫu (Instance Normalization trên 2 chiều tần số và thời gian)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + self.norm_eps)
        # Cắt khung và thêm chiều channel đơn (1)
        return self._crop(x, self._offsets(x.shape[-1], x.shape[0], x.device)).unsqueeze(1)


class SpecAugment(nn.Module):
    """
    Tăng cường dữ liệu trên phổ (SpecAugment):
        - Che ngẫu nhiên dải tần số (Frequency Masking)
        - Che ngẫu nhiên khoảng thời gian (Time Masking)
        
    Chỉ kích hoạt trong chế độ huấn luyện (training mode).
    """
    def __init__(self, freq_masks: int = 2, freq_width: int = 16, time_masks: int = 2, time_width: int = 24) -> None:
        super().__init__()
        if freq_masks < 0 or time_masks < 0 or freq_width < 0 or time_width < 0:
            raise ValueError("mask counts and widths must be non-negative")
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    @staticmethod
    def _zero_band(x: Tensor, sample: int, dim: int, width: int) -> None:
        """Gán giá trị 0 cho một dải liên tục trên chiều dim (tần số hoặc thời gian)"""
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
        """Áp dụng mặt nạ ngẫu nhiên lên batch phổ [B, 1, F, T]"""
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x
        x = x.clone()
        for sample in range(x.shape[0]):
            for _ in range(self.freq_masks):
                self._zero_band(x, sample, dim=2, width=self.freq_width)
            for _ in range(self.time_masks):
                self._zero_band(x, sample, dim=3, width=self.time_width)
        return x
