"""
Comparative Vision Transformer Baselines (0 - 25M Parameters).

References:
    - DeiT: Touvron et al., "Training data-efficient image transformers & distillation
      through attention", ICML 2021. arXiv:2012.12877.
    - ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
      Recognition at Scale", ICLR 2021. arXiv:2010.11929.
    - Audio Spectrogram Transformer (AST): Gong et al., "AST: Audio Spectrogram Transformer",
      Interspeech 2021. arXiv:2104.01778.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch import Tensor

from src.frontend import LogMel, SpecAugment, rms_normalize
from src.models.vit_bias import ViTSBias, load_ced_tiny


class DeiTTinyBaseline(nn.Module):
    """
    Standard DeiT-Tiny Baseline (~5.7M parameters).
    Architecture:
        - 12 Transformer blocks, embed_dim=192, 3 attention heads, MLP dim=768
        - Standard 1D Absolute Position Embedding (ImageNet pre-trained)
        - No audio-specific 2D inductive bias (demonstrates the gap between generic ViT and CED)

    Reference:
        Touvron et al., "Training data-efficient image transformers & distillation
        through attention", ICML 2021.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            img_size=(freq_bins, frames),
            drop_rate=dropout,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        # Standardize RMS to 0.07925
        wave_norm = rms_normalize(waveform)
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())
        mel_aug = self.spec_augment(mel)
        return self.backbone(mel_aug)


class ViTSSmallBaseline(nn.Module):
    """
    DeiT / ViT Small Baseline (~22.0M parameters, within 0-25M limit).
    Architecture:
        - 12 Transformer blocks, embed_dim=384, 6 attention heads, MLP dim=1536
        - 1D Position Embedding, ImageNet pre-trained
        - Evaluates if simply scaling ViT capacity without audio inductive bias can match CED-Tiny

    Reference:
        Touvron et al., "Training data-efficient image transformers", ICML 2021.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            img_size=(freq_bins, frames),
            drop_rate=dropout,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        wave_norm = rms_normalize(waveform)
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())
        mel_aug = self.spec_augment(mel)
        return self.backbone(mel_aug)


class CEDTinyBaseline(nn.Module):
    """
    Standard Single-Model CED-Tiny (~5.42M parameters).
    Architecture:
        - 12 Transformer blocks, embed_dim=192, 3 attention heads, MLP dim=768
        - 2D Gated Relative Position Bias (BiasAttention)
        - Pre-trained on AudioSet (Consistent Ensemble Distillation)

    Reference:
        Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
        ICASSP 2024. arXiv:2308.11957.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.vit = ViTSBias(
            num_classes=num_classes,
            freq_bins=freq_bins,
            frames=frames,
            pretrained=False,
            dropout=dropout,
            use_bias=True,
            time_major=False,
        )
        load_ced_tiny(self.vit)

    def forward(self, waveform: Tensor) -> Tensor:
        wave_norm = rms_normalize(waveform)
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())
        mel_aug = self.spec_augment(mel)
        return self.vit(mel_aug)
