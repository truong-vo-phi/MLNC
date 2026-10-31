"""
Single-Backbone CED-Tiny with Asymmetric HydraLoRA Experts (CED-HydraLoRA-MoE).

References:
    - HydraLoRA: Tian et al., "HydraLoRA: An Asymmetric LoRA Architecture for Efficient
      Fine-Tuning", NeurIPS 2024 Oral. arXiv:2404.19245.
    - MoLE: Wu et al., "Mixture of LoRA Experts", ICLR 2024. arXiv:2404.13628.
    - LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
      ICLR 2022. arXiv:2106.09685.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.frontend import AcousticFeatureExtractor, LogMel, SpecAugment, rms_normalize
from src.models.vit_bias import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
from src.router import SelfContainedSpectralTokenRouter


class HydraLinear(nn.Module):
    """
    Asymmetric HydraLoRA Linear Layer.
    Formula:
        y = W_0 * x + (scaling / r) * [ (1 - alpha) * B_low + alpha * B_high ] * (A * x)

    References:
        - Tian et al., "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning",
          NeurIPS 2024 Oral. arXiv:2404.19245.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.scaling = lora_alpha / r
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Shared down-projection matrix A (learns general domain manifold)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        # Asymmetric specialized up-projection matrices B (learns regime-specific residual)
        self.lora_B_low = nn.Parameter(torch.empty(out_features, r))
        self.lora_B_high = nn.Parameter(torch.empty(out_features, r))

        # Weight initialization: Kaiming uniform for A, zeros for B (exact identity warm-start)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_low)
        nn.init.zeros_(self.lora_B_high)

        self.current_alpha: Tensor | None = None

    def set_alpha(self, alpha: Tensor) -> None:
        self.current_alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        if self.current_alpha is None:
            return base_out

        # x: [..., in_features]
        alpha = self.current_alpha
        # A * x
        h_low_rank = F.linear(x, self.lora_A)  # [..., r]

        # Blend up-projections
        delta_low = F.linear(h_low_rank, self.lora_B_low)
        delta_high = F.linear(h_low_rank, self.lora_B_high)

        if alpha.ndim == 2 and base_out.ndim == 3:
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]

        delta_fused = (1.0 - alpha) * delta_low + alpha * delta_high
        return base_out + self.scaling * delta_fused


class CEDHydraLoRAMoE(nn.Module):
    """
    Single-Backbone CED-Tiny with Asymmetric HydraLoRA Experts (~5.58M parameters).
    - Preserves all 5.42M weights of CED-Tiny.
    - Injects rank-16 HydraLoRA into MLP layers of Blocks 6 through 11.
    - Self-Contained Spectral-Token Router for non-oracle inference.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        target_blocks: tuple[int, ...] = (6, 7, 8, 9, 10, 11),
        r: int = 16,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_blocks = target_blocks
        self.num_classes = num_classes

        # 1. Frontends
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.acoustic_extractor = AcousticFeatureExtractor()

        # 2. Backbone base model
        self.vit_base = ViTSBias(
            num_classes=num_classes,
            freq_bins=freq_bins,
            frames=frames,
            pretrained=False,
            dropout=dropout,
            use_bias=True,
            time_major=False,
        )
        load_ced_tiny(self.vit_base)

        # 3. Router
        self.router = SelfContainedSpectralTokenRouter(
            spectral_dim=6,
            token_dim=192,
            token_proj_dim=16,
            hidden_dim=32,
        )

        # 4. Inject HydraLinear layers
        self.hydra_layers: list[HydraLinear] = []
        for block_idx in self.target_blocks:
            block = self.vit_base.vit.blocks[block_idx]
            fc1 = HydraLinear(block.mlp.fc1, r=r, lora_alpha=lora_alpha)
            fc2 = HydraLinear(block.mlp.fc2, r=r, lora_alpha=lora_alpha)
            block.mlp.fc1 = fc1
            block.mlp.fc2 = fc2
            self.hydra_layers.extend([fc1, fc2])

        # 5. Dual Classifier Heads
        original_head = self.vit_base.vit.head
        self.head_low = copy.deepcopy(original_head)
        self.head_high = copy.deepcopy(original_head)

    def forward(
        self,
        waveform: Tensor,
        return_dict: bool = False,
    ) -> dict[str, Tensor] | Tensor:
        wave_norm = rms_normalize(waveform)
        spectral_feat = self.acoustic_extractor(wave_norm)

        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())
        mel_aug = self.spec_augment(mel)

        vit = self.vit_base.vit
        x = vit.patch_embed(mel_aug[..., : self.vit_base.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))

        self.vit_base.shared.value = None

        # Execute Blocks 0 through 5 (early acoustic representations)
        for i in range(6):
            x = vit.blocks[i](x)

        # Extract Block 5 token representation for router (mid-layer acoustic context)
        mid_cls = x[:, 0]
        alpha, gates = self.router(spectral_feat, mid_cls)

        # Pass alpha to all HydraLoRA layers before executing Blocks 6 through 11
        for layer in self.hydra_layers:
            layer.set_alpha(alpha)

        # Execute blocks 6 through 11 with HydraLoRA adaptation active
        for i in range(6, 12):
            x = vit.blocks[i](x)

        for layer in self.hydra_layers:
            layer.set_alpha(None)

        x = vit.norm(x)
        cls_final = x[:, 0]

        logits_low = self.head_low(cls_final)
        logits_high = self.head_high(cls_final)
        logits_fused = (1.0 - alpha) * logits_low + alpha * logits_high

        if return_dict:
            return {
                "logits": logits_fused,
                "logits_low": logits_low,
                "logits_high": logits_high,
                "alpha": alpha,
                "gates": gates,
            }
        return logits_fused
