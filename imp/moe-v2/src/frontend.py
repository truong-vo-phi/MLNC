"""
Frontend Audio Processing and Acoustic Feature Extraction for Single-Backbone MoE.

Pure PyTorch Implementation: 100% numerically identical to CED MelSpectrogram
without requiring compiled C++ torchaudio shared library extensions.

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
    - SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method
      for Automatic Speech Recognition", Interspeech 2019. arXiv:1904.08779.
    - Spectral Flatness & Acoustic Features: Peeters, "A large set of audio features
      for sound description (similarity and classification) in the CUIDADO project", 2004.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RMS_EPS = 1e-8


def rms_normalize(waveform: Tensor, target_rms: float = 0.07925) -> Tensor:
    """
    Standardize the energy level of the waveform to target_rms (0.07925).
    Formula: x_norm = x * (target_rms / (sqrt(mean(x^2)) + eps))
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
    """Construct standard HTK Mel Filterbank matrix identical to torchaudio."""
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
    Pure PyTorch Log-Mel Spectrogram Frontend matching the CED specification:
    - Sample rate: 16kHz
    - FFT: 512, Win length: 400, Hop length: 160
    - Mel bins: 64, Frequency range: 20Hz - 8000Hz
    - Instance normalization over (time, frequency)

    Reference:
        Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
        ICASSP 2024. arXiv:2308.11957.
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
        if self.pad:
            waveform = F.pad(waveform, (0, self.pad))
        # STFT computation
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
        # Mel matrix multiplication: [n_mels, n_freqs] @ [B, n_freqs, Frames] -> [B, n_mels, Frames]
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
        x = self.logmel(waveform)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + self.norm_eps)
        return self._crop(x, self._offsets(x.shape[-1], x.shape[0], x.device)).unsqueeze(1)


class SpecAugment(nn.Module):
    """
    Frequency and Time Masking SpecAugment.

    Reference:
        Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR",
        Interspeech 2019. arXiv:1904.08779.
    """
    def __init__(self, freq_masks: int = 2, freq_width: int = 8, time_masks: int = 2, time_width: int = 24) -> None:
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


class AcousticFeatureExtractor(nn.Module):
    """
    Extract 6 intrinsic acoustic and spectral morphology features from waveform n_hat:
    1. Log-RMS energy (raw scale proxy)
    2. Crest factor (peak amplitude to RMS ratio)
    3. Frame-wise energy kurtosis (captures sharp phoneme transients from speech leakage)
    4. Spectral Flatness (Wiener entropy: ratio of geometric mean to arithmetic mean)
    5. Low-band to Speech-band energy ratio (<300 Hz vs 300-3400 Hz)
    6. High-band to Speech-band energy ratio (>3400 Hz vs 300-3400 Hz)

    These features enable internal self-contained routing without needing the clean signal or Oracle SNR!
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 160) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, waveform: Tensor) -> Tensor:
        B, T_len = waveform.shape
        eps = 1e-8

        # 1. Log-RMS
        rms = waveform.square().mean(dim=-1).clamp(min=eps).sqrt()
        log_rms = torch.log(rms + eps).unsqueeze(-1)  # [B, 1]

        # 2. Crest factor: peak / rms
        peak = waveform.abs().max(dim=-1).values
        crest_factor = (peak / (rms + eps)).unsqueeze(-1)  # [B, 1]

        # Short-time Fourier transform for spectral morphology
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )  # [B, F, Frames]
        power = stft.abs().square() + eps  # [B, F, Frames]

        # 3. Frame-wise energy kurtosis
        frame_energy = power.sum(dim=1)  # [B, Frames]
        mean_fe = frame_energy.mean(dim=-1, keepdim=True)
        std_fe = frame_energy.std(dim=-1, keepdim=True) + eps
        kurtosis = (((frame_energy - mean_fe) / std_fe) ** 4).mean(dim=-1, keepdim=True)  # [B, 1]

        # 4. Spectral Flatness: exp(mean(log(power))) / mean(power) across frequency bins
        log_power = torch.log(power)
        geom_mean = torch.exp(log_power.mean(dim=1))
        arith_mean = power.mean(dim=1)
        flatness = (geom_mean / (arith_mean + eps)).mean(dim=-1, keepdim=True)  # [B, 1]

        # 5 & 6. Sub-band energy ratios
        band_low = power[:, :10, :].sum(dim=(1, 2)) + eps
        band_speech = power[:, 10:109, :].sum(dim=(1, 2)) + eps
        band_high = power[:, 109:, :].sum(dim=(1, 2)) + eps

        ratio_low_speech = torch.log(band_low / band_speech).unsqueeze(-1)  # [B, 1]
        ratio_high_speech = torch.log(band_high / band_speech).unsqueeze(-1)  # [B, 1]

        features = torch.cat(
            [log_rms, crest_factor, kurtosis, flatness, ratio_low_speech, ratio_high_speech],
            dim=-1,
        )
        return features
