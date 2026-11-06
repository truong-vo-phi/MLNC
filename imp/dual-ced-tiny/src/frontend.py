"""
Audio Frontend Processing Module for Dual-CED-Tiny.
Extracts 64-Mel Log-Filterbank Spectrograms [B, 1, 64, 398] and Acoustic Metrics from 16kHz waveforms,
with optional SpecAugment time/frequency masking during training.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FrontendConfig(NamedTuple):
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


class SpecAugment(nn.Module):
    """
    SpecAugment on Z-score normalized Log-Mel spectrograms [B, 1, F, T] or [B, F, T].
    Applies frequency and time masking with 0.0 (mean value) only when `self.training` is True.
    """

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 8,
        time_masks: int = 2,
        time_width: int = 24,
    ) -> None:
        super().__init__()
        self.freq_masks = int(freq_masks)
        self.freq_width = int(freq_width)
        self.time_masks = int(time_masks)
        self.time_width = int(time_width)

    def forward(self, spec: Tensor) -> Tensor:
        if not self.training or (self.freq_masks <= 0 and self.time_masks <= 0):
            return spec

        squeeze_channel = False
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
            squeeze_channel = True

        out = spec.clone()
        b, _, n_mels, n_frames = out.shape
        device = out.device

        for i in range(b):
            for _ in range(self.freq_masks):
                if self.freq_width > 0 and n_mels > 1:
                    w = int(torch.randint(1, min(self.freq_width + 1, n_mels), (1,), device=device).item())
                    f0 = int(torch.randint(0, max(1, n_mels - w + 1), (1,), device=device).item())
                    out[i, :, f0 : f0 + w, :] = 0.0

            for _ in range(self.time_masks):
                if self.time_width > 0 and n_frames > 1:
                    w = int(torch.randint(1, min(self.time_width + 1, n_frames), (1,), device=device).item())
                    t0 = int(torch.randint(0, max(1, n_frames - w + 1), (1,), device=device).item())
                    out[i, :, :, t0 : t0 + w] = 0.0

        return out.squeeze(1) if squeeze_channel else out


class LogMelFrontend(nn.Module):
    """
    LogMel Spectrogram Extractor for CED-Tiny:
    1. STFT with Hann window (win=400, hop=160, n_fft=512).
    2. Triangular Mel Filterbank (64 Mel bins, 20Hz - 8000Hz).
    3. Log-compression: log(mel + eps).
    4. Instance Z-score normalization per sample across (F, T).
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
        log_eps: float = 1.0e-6,
        norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_frames = num_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps

        # Register Hann window
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

        # Create Mel filterbank matrix
        mel_fb = self._create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_fb", mel_fb)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float
    ) -> Tensor:
        num_bins = n_fft // 2 + 1
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mels = torch.linspace(mel_min, mel_max, n_mels + 2)
        hzs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        bins = torch.floor((n_fft + 1) * hzs / sr).long()

        fb = torch.zeros(n_mels, num_bins)
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m > f_m_minus:
                fb[m - 1, f_m_minus:f_m] = (torch.arange(f_m_minus, f_m) - f_m_minus) / float(f_m - f_m_minus)
            if f_m_plus > f_m:
                fb[m - 1, f_m:f_m_plus] = (f_m_plus - torch.arange(f_m, f_m_plus)) / float(f_m_plus - f_m)

        return fb  # Shape: (n_mels, num_bins)

    def forward(self, wave: Tensor) -> Tensor:
        """
        Input:
            wave: (B, T_samples) or (T_samples,)
        Output:
            spec: (B, 1, n_mels, num_frames)
        """
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)

        # CED official frontend: center=False with (n_fft - win_length) zero padding at the end.
        # For a 4.0 s / 16 kHz clip this yields exactly 398 frames, matching mispeech/ced-tiny.
        if self.win_length < self.n_fft:
            wave = F.pad(wave, (0, self.n_fft - self.win_length))

        stft = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (B, F_bins, T_frames)

        mag = torch.abs(stft) ** 2  # Power spectrum
        mel = torch.matmul(self.mel_fb, mag)  # (B, n_mels, T_frames)
        log_mel = torch.log(mel + self.log_eps)

        # Align time frames to num_frames
        curr_frames = log_mel.shape[-1]
        if curr_frames < self.num_frames:
            log_mel = F.pad(log_mel, (0, self.num_frames - curr_frames))
        elif curr_frames > self.num_frames:
            log_mel = log_mel[..., : self.num_frames]

        # Instance Z-score normalization per sample
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + self.norm_eps
        norm_mel = (log_mel - mean) / std

        return norm_mel.unsqueeze(1)  # (B, 1, n_mels, num_frames)


class AcousticFeatureExtractor(nn.Module):
    """Extract the six waveform cues used by the self-contained MoE-v2 router.

    Input is a batch of 16 kHz waveforms ``[B, samples]``. The output is
    ``[B, 6]``: log-RMS, crest factor, frame-energy kurtosis, spectral
    flatness, low-to-speech-band energy ratio, and high-to-speech-band ratio.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 160) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"router waveform must be [B, samples], got {tuple(waveform.shape)}")
        waveform = waveform.float()
        eps = 1e-8
        rms = waveform.square().mean(dim=-1).clamp_min(eps).sqrt()
        log_rms = torch.log(rms + eps).unsqueeze(-1)
        crest = (waveform.abs().amax(dim=-1) / (rms + eps)).unsqueeze(-1)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )
        power = stft.abs().square() + eps
        frame_energy = power.sum(dim=1)
        mean_energy = frame_energy.mean(dim=-1, keepdim=True)
        std_energy = frame_energy.std(dim=-1, keepdim=True).clamp_min(eps)
        kurtosis = (((frame_energy - mean_energy) / std_energy) ** 4).mean(dim=-1, keepdim=True)

        geometric_mean = power.log().mean(dim=1).exp()
        arithmetic_mean = power.mean(dim=1)
        flatness = (geometric_mean / (arithmetic_mean + eps)).mean(dim=-1, keepdim=True)

        band_low = power[:, :10, :].sum(dim=(1, 2)) + eps
        band_speech = power[:, 10:109, :].sum(dim=(1, 2)) + eps
        band_high = power[:, 109:, :].sum(dim=(1, 2)) + eps
        low_ratio = torch.log(band_low / band_speech).unsqueeze(-1)
        high_ratio = torch.log(band_high / band_speech).unsqueeze(-1)
        return torch.cat([log_rms, crest, kurtosis, flatness, low_ratio, high_ratio], dim=-1)
