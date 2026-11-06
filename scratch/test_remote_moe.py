import torch
from pathlib import Path

# Fix frontend.py to avoid torchaudio
frontend_path = Path("/marimo/src/frontend.py")
frontend_code = """
from __future__ import annotations
import math
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FrontendOutput(NamedTuple):
    features: Tensor
    lengths: Tensor | None = None

def create_mel_filterbank(sr=16000, n_fft=1024, n_mels=128, f_min=20.0, f_max=8000.0) -> Tensor:
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
    def __init__(self, sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128, f_min=20.0, f_max=8000.0, num_frames=398, log_eps=1.0e-6, norm_eps=1.0e-5):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_frames = num_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps
        self.register_buffer('window', torch.hann_window(win_length))
        self.register_buffer('mel_fb', create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max))

    def forward(self, waveform: Tensor) -> FrontendOutput:
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=True, return_complex=True)
        power = stft.abs().square()
        mel = torch.matmul(self.mel_fb, power)
        log_mel = torch.log(mel + self.log_eps)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True)
        norm_mel = (log_mel - mean) / (std + self.norm_eps)
        cur_frames = norm_mel.shape[-1]
        if cur_frames < self.num_frames:
            norm_mel = F.pad(norm_mel, (0, self.num_frames - cur_frames))
        elif cur_frames > self.num_frames:
            norm_mel = norm_mel[..., : self.num_frames]
        return FrontendOutput(features=norm_mel.unsqueeze(1))

class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, freq_width=16, time_masks=2, time_width=24):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x
        b, c, f, t = x.shape
        out = x.clone()
        for _ in range(self.freq_masks):
            w = torch.randint(0, min(self.freq_width, f), (b,), device=x.device)
            f0 = torch.randint(0, f, (b,), device=x.device)
            for i in range(b):
                if w[i] > 0:
                    start = min(f0[i].item(), f - w[i].item())
                    out[i, :, start : start + w[i].item(), :] = 0.0
        for _ in range(self.time_masks):
            w = torch.randint(0, min(self.time_width, t), (b,), device=x.device)
            t0 = torch.randint(0, t, (b,), device=x.device)
            for i in range(b):
                if w[i] > 0:
                    start = min(t0[i].item(), t - w[i].item())
                    out[i, :, :, start : start + w[i].item()] = 0.0
        return out
"""
frontend_path.write_text(frontend_code, encoding="utf-8")
print("frontend.py updated successfully.")

from src.models.ced_moe_split import (
    CEDSingleMoESplitFFN,
    AcousticFeatureExtractor,
    SelfContainedSpectralTokenRouter,
    SplitMoEFFN,
)

print("Torch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Test AcousticFeatureExtractor (6 features)
extractor = AcousticFeatureExtractor().to(device)
waveform = torch.randn(2, 64000, device=device)
f_spec = extractor(waveform)
print("f_spec shape (should be [2, 6]):", f_spec.shape)
print("f_spec values sample:\n", f_spec[0].cpu().tolist())

# 2. Test Full CED-SingleMoE-SplitFFN
model = CEDSingleMoESplitFFN(
    num_classes=36,
    load_pretrained=True,
    pretrained_path="/marimo/pretrained/model.safetensors"
).to(device)

params = model.get_num_params()
print("Model params:", params)

out = model.forward_full(waveform)
print("Logits shape:", out["logits"].shape)
print("Alpha shape:", out["alpha"].shape)
print("Alpha values:", out["alpha"].squeeze().cpu().tolist())
print("Pretrained loaded tensors:", model.pretrained_loaded_tensors)
