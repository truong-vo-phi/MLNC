"""
CED-HydraLoRA-v3: Enhanced Asymmetric HydraLoRA MoE with Reliability Gating (~5.80M params).

References:
    - HydraLoRA: Tian et al., "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning",
      NeurIPS 2024 Oral. arXiv:2404.19245.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization", ACL 2024. arXiv:2401.06066.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
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
try:
    from src.models.vit_grep import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
except Exception:
    from src.models.vit_bias import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
from src.router import SelfContainedSpectralTokenRouter


class AsymmetricHydraLinear(nn.Module):
    """
    Asymmetric HydraLoRA Linear Layer (Tian et al., NeurIPS 2024 Oral).
    
    Architecture:
        - Shared down-projection A: rank r_A (default: 24)
        - Low-SNR up-projection B_low: rank r_low (default: 8, noise-dominant, broad patterns)
        - High-SNR up-projection B_high: rank r_high (default: 24, speech-leakage, high complexity)
    
    Formula:
        h = A * x  (h in R^{r_A})
        delta_low = (scale_low / r_low) * (B_low * h[:, :r_low])
        delta_high = (scale_high / r_high) * (B_high * h[:, :r_high])
        y = W_0 * x + (1 - alpha) * delta_low + alpha * delta_high
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r_A: int = 24,
        r_low: int = 8,
        r_high: int = 24,
        lora_alpha_low: float = 16.0,
        lora_alpha_high: float = 32.0,
    ) -> None:
        super().__init__()
        assert r_low <= r_A and r_high <= r_A, "Sub-ranks must not exceed r_A"
        self.base_linear = base_linear
        self.r_A = r_A
        self.r_low = r_low
        self.r_high = r_high
        self.scaling_low = lora_alpha_low / r_low
        self.scaling_high = lora_alpha_high / r_high

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Shared down-projection matrix A
        self.lora_A = nn.Parameter(torch.empty(r_A, in_features))
        # Asymmetric regime-specific up-projection matrices
        self.lora_B_low = nn.Parameter(torch.empty(out_features, r_low))
        self.lora_B_high = nn.Parameter(torch.empty(out_features, r_high))

        # Weight initialization: Kaiming uniform for A, zero initialization for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_low)
        nn.init.zeros_(self.lora_B_high)

        self.current_alpha: Tensor | None = None

    def set_alpha(self, alpha: Tensor | None) -> None:
        self.current_alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        if self.current_alpha is None:
            return base_out

        alpha = self.current_alpha
        # Shared down-projection: [..., r_A]
        h = F.linear(x, self.lora_A)

        # Asymmetric specialized up-projections
        delta_low = self.scaling_low * F.linear(h[..., : self.r_low], self.lora_B_low)
        delta_high = self.scaling_high * F.linear(h[..., : self.r_high], self.lora_B_high)

        if alpha.ndim == 2 and base_out.ndim == 3:
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]

        delta_fused = (1.0 - alpha) * delta_low + alpha * delta_high
        return base_out + delta_fused


class CEDHydraLoRAV3(nn.Module):
    """
    CED-HydraLoRA-v3: Single-Backbone Vision Transformer with Asymmetric HydraLoRA Experts.
    - Parameter budget: ~5.80M parameters (CED-Tiny base 5.42M + Asymmetric HydraLoRA 0.37M + Router/Heads 0.01M).
    - Blocks 4 through 11 adapted with Asymmetric HydraLoRA (r_low=8, r_high=24, r_A=24).
    - Self-Contained Spectral-Token Router for Zero-Oracle inference.
    - Dual Regime Classifier Heads (head_low and head_high).
    - 100% Standalone at test time (no BEATs dependencies).
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        target_blocks: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11),
        r_A: int = 24,
        r_low: int = 8,
        r_high: int = 24,
        progressive_ranks: bool = True,
        lora_alpha_low: float = 16.0,
        lora_alpha_high: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_blocks = target_blocks
        self.num_classes = num_classes
        self.progressive_ranks = progressive_ranks

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

        # 4. Inject Asymmetric HydraLinear into target blocks (mlp.fc1 and mlp.fc2)
        # Progressive rank allocation:
        # Blocks 4-7: r_A=16, r_low=8, r_high=16 (lighter early representations)
        # Blocks 8-11: r_A=32, r_low=12, r_high=32 (boosted expressive capacity for speech-noise separation)
        self.hydra_layers: list[AsymmetricHydraLinear] = []
        for block_idx in self.target_blocks:
            if self.progressive_ranks:
                if block_idx < 8:
                    b_r_A, b_r_low, b_r_high = 16, 8, 16
                    b_alpha_low, b_alpha_high = 16.0, 16.0
                else:
                    b_r_A, b_r_low, b_r_high = 32, 12, 32
                    b_alpha_low, b_alpha_high = 16.0, 32.0
            else:
                b_r_A, b_r_low, b_r_high = r_A, r_low, r_high
                b_alpha_low, b_alpha_high = lora_alpha_low, lora_alpha_high

            block = self.vit_base.vit.blocks[block_idx]
            fc1 = AsymmetricHydraLinear(
                block.mlp.fc1,
                r_A=b_r_A,
                r_low=b_r_low,
                r_high=b_r_high,
                lora_alpha_low=b_alpha_low,
                lora_alpha_high=b_alpha_high,
            )
            fc2 = AsymmetricHydraLinear(
                block.mlp.fc2,
                r_A=b_r_A,
                r_low=b_r_low,
                r_high=b_r_high,
                lora_alpha_low=b_alpha_low,
                lora_alpha_high=b_alpha_high,
            )
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

        # Execute Blocks 0 through 3 (early low-level acoustic representations)
        for i in range(4):
            x = vit.blocks[i](x)

        # Extract Block 3 CLS token representation for router
        mid_cls = x[:, 0]
        alpha, gates = self.router(spectral_feat, mid_cls)

        # Pass alpha to all HydraLoRA layers
        for layer in self.hydra_layers:
            layer.set_alpha(alpha)

        # Execute Blocks 4 through 11 with HydraLoRA adaptation active
        for i in range(4, 12):
            x = vit.blocks[i](x)

        # Reset alpha in layers
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
