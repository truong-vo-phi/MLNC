"""
Acoustic Frontend and Physics Feature Extractor for CED-SingleMoE-SplitFFN.

Components:
1. rms_normalize: RMS Energy Normalization to sigma_0 = 0.07925 (~ -22.02 dBFS).
2. AcousticFeatureExtractor:
   - STFT linear power spectrum P(f, t) [B, 257, 401] from 64,000 samples (16 kHz, 4s).
   - 6 Physics & Spectral morphology features (f_1 to f_6):
     * f_1: Log-RMS energy level
     * f_2: Crest factor (peak / RMS) - impulse burst detection
     * f_3: Temporal Kurtosis of frame energy (detects sporadic speech leakage transients from FRCRN)
     * f_4: Spectral Flatness (Wiener entropy: geom_mean / arith_mean)
     * f_5: Low-band to speech-band energy ratio (<300 Hz vs 300-3400 Hz)
     * f_6: High-band to speech-band energy ratio (>3400 Hz vs 300-3400 Hz)
3. LogMel: HTK Mel Filterbank (64 mels, 398 frames) with per-sample z-score normalization.
4. SpecAugment: Frequency and Time band masking.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RMS_EPS = 1e-8
TARGET_RMS = 0.07925  # Reference RMS energy level (~ -22.02 dBFS)


def rms_normalize(waveform: Tensor, target_rms: float = TARGET_RMS) -> Tensor:
    """
    Normalizes waveform RMS energy to target_rms = 0.07925.
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
    """Creates HTK Mel filterbank matrix matching CED standard."""
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


class AcousticFeatureExtractor(nn.Module):
    """
    Extracts 6 physical, spectral, and statistical features directly from audio waveform:
    Input: Normalized waveform n_hat of 64,000 samples (4.0s at 16 kHz).
    Linear STFT:
        - n_fft = 512 -> 257 linear frequency bins (0 to 8000 Hz, delta_f = 31.25 Hz).
        - hop_length = 160 -> 10 ms hop.
        - win_length = 512, center=True -> 401 time frames.
        - Power spectrum P(f, t) [B, 257, 401].

    6 Acoustic Features [f_1, f_2, f_3, f_4, f_5, f_6]:
        1. f_1 (Log-RMS): Total signal energy level.
        2. f_2 (Crest Factor): Peak amplitude over RMS.
        3. f_3 (Temporal Kurtosis): 4th moment of frame energy across time.
           Stationary noise has Kurtosis ~ 3. FRCRN speech leakage creates sharp bursts (>> 3).
        4. f_4 (Spectral Flatness / Wiener entropy): geom_mean / arith_mean.
           Near 1 for flat noise; << 1 for harmonic speech residuals.
        5. f_5 (Low-band to Speech-band Ratio): log(E_<300Hz / E_300-3400Hz).
        6. f_6 (High-band to Speech-band Ratio): log(E_>3400Hz / E_300-3400Hz).
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 160) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def compute_stft_power(self, waveform: Tensor) -> Tensor:
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
        B, T_len = waveform.shape
        eps = 1e-8

        # 1. f_1: Log-RMS
        rms = waveform.square().mean(dim=-1).clamp(min=eps).sqrt()
        log_rms = torch.log(rms + eps).unsqueeze(-1)  # [B, 1]

        # 2. f_2: Crest Factor (peak / rms)
        peak = waveform.abs().max(dim=-1).values
        crest_factor = (peak / (rms + eps)).unsqueeze(-1)  # [B, 1]

        # Power spectrum P(f, t)
        power = self.compute_stft_power(waveform) + eps

        # 3. f_3: Temporal Kurtosis
        frame_energy = power.sum(dim=1)  # [B, 401]
        mean_fe = frame_energy.mean(dim=-1, keepdim=True)
        std_fe = frame_energy.std(dim=-1, keepdim=True) + eps
        kurtosis = (((frame_energy - mean_fe) / std_fe) ** 4).mean(dim=-1, keepdim=True)  # [B, 1]

        # 4. f_4: Spectral Flatness (Wiener entropy)
        log_power = torch.log(power)
        geom_mean = torch.exp(log_power.mean(dim=1))  # [B, 401]
        arith_mean = power.mean(dim=1)                # [B, 401]
        flatness = (geom_mean / (arith_mean + eps)).mean(dim=-1, keepdim=True)  # [B, 1]

        # 5 & 6. Sub-band energy ratios
        band_low = power[:, :10, :].sum(dim=(1, 2)) + eps        # <300 Hz (bins 0..9)
        band_speech = power[:, 10:109, :].sum(dim=(1, 2)) + eps   # 300-3400 Hz (bins 10..108)
        band_high = power[:, 109:, :].sum(dim=(1, 2)) + eps      # >3400 Hz (bins 109..256)

        ratio_low_speech = torch.log(band_low / band_speech).unsqueeze(-1)    # [B, 1]
        ratio_high_speech = torch.log(band_high / band_speech).unsqueeze(-1)  # [B, 1]

        features = torch.cat(
            [log_rms, crest_factor, kurtosis, flatness, ratio_low_speech, ratio_high_speech],
            dim=-1,
        )  # [B, 6]
        return features


class LogMel(nn.Module):
    """
    Log-Mel Spectrogram Frontend:
    - 16 kHz sample rate, n_fft=512, hop_length=160, win_length=400
    - 64 Mel filterbanks, 398 frames
    - Per-sample z-score normalization across frequency and time
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
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = stft.abs().square()
        mel = torch.matmul(self.mel_fb, power)
        return (mel + self.log_eps).log()

    def forward(self, waveform: Tensor) -> Tensor:
        """Computes per-sample z-score normalized Log-Mel spectrogram [B, 1, 64, 398]."""
        mel = self.logmel(waveform)  # [B, 64, T]
        # Crop or pad to target num_frames
        if mel.shape[-1] > self.num_frames:
            mel = mel[:, :, :self.num_frames]
        elif mel.shape[-1] < self.num_frames:
            mel = F.pad(mel, (0, self.num_frames - mel.shape[-1]))

        # Per-sample instance z-score normalization
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True) + self.norm_eps
        normed = (mel - mean) / (std * 2.0)
        return normed.unsqueeze(1)  # [B, 1, 64, 398]


class SpecAugment(nn.Module):
    """SpecAugment frequency and time masking."""
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

