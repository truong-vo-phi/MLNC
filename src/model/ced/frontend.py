from __future__ import annotations

import torch
from torch import nn
from torchaudio import transforms


class MelFrontend(nn.Module):
    """Waveform (B, samples) -> log-mel dB (B, n_mels, T)."""

    def __init__(
        self,
        n_mels=64,
        sample_rate=16000,
        n_fft=512,
        win_size=512,
        hop_size=160,
        f_min=0,
        f_max=8000,
        center=True,
        top_db=120.0,
    ):
        super().__init__()
        self.mel = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=center,
        )
        self.to_db = transforms.AmplitudeToDB(top_db=top_db)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            return self.to_db(self.mel(waveform.float()).unsqueeze(1)).squeeze(1)
