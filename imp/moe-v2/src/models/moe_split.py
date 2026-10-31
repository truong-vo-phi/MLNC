"""
Single-Backbone CED-Tiny with Split-FFN MoE and Shared Expert (CED-SingleMoE-SplitFFN).

References:
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts",
      NeurIPS 2021. arXiv:2106.05974 (Section 3.2: Last-K Layer MoE).
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts",
      ICLR 2024. arXiv:2308.00951 (Differentiable Soft Mixture Routing).
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization
      in Mixture-of-Experts Language Models", ACL 2024. arXiv:2401.06066
      (Shared-Expert + Domain-Routed Experts architecture).
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.frontend import AcousticFeatureExtractor, LogMel, SpecAugment, rms_normalize
from src.models.vit_bias import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
from src.router import SelfContainedSpectralTokenRouter


class SplitMoEFFN(nn.Module):
    """
    Split-FFN with Shared Expert and Dual Routed Experts (Low-SNR & High-SNR).
    Formula:
        y = FFN_shared(x) + (1 - alpha) * FFN_low(x) + alpha * FFN_high(x)

    References:
        - DeepSeekMoE (Dai et al., ACL 2024): 1 Shared Expert + E Routed Experts.
        - Soft MoE (Puigcerver et al., ICLR 2024): Continuous interpolation via alpha.
    """
    def __init__(self, base_mlp: nn.Module) -> None:
        super().__init__()
        # 1 Shared Expert + 2 Routed Experts (Low and High)
        self.ffn_shared = copy.deepcopy(base_mlp)
        self.ffn_low = copy.deepcopy(base_mlp)
        self.ffn_high = copy.deepcopy(base_mlp)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        # x: [B, Tokens, C]
        # alpha: [B, 1] or [B, 1, 1]
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]

        out_shared = self.ffn_shared(x)
        out_low = self.ffn_low(x)
        out_high = self.ffn_high(x)

        # Soft convex routing combination with normalized scale:
        routed_out = (1.0 - alpha) * out_low + alpha * out_high
        return 0.5 * out_shared + 0.5 * routed_out


class CEDSingleMoESplitFFN(nn.Module):
    """
    Single-Backbone CED-Tiny MoE Architecture (~6.02M parameters).

    - Blocks 0 to 9: 100% shared standard Transformer blocks with 2D GREP BiasAttention.
    - Blocks 10 & 11: Split-FFN containing 1 Shared Expert + 2 Routed Experts (Low & High).
    - Classifier: Dual-Head (Head_Low and Head_High) blended via alpha.
    - Self-Contained Spectral-Token Router: Uses physical features from AcousticFeatureExtractor
      and intermediate token representation from Block 9.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        split_blocks: tuple[int, ...] = (10, 11),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.split_blocks = split_blocks
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

        # 4. Replace MLPs in split blocks with SplitMoEFFN
        self.moe_blocks: dict[int, SplitMoEFFN] = {}
        for block_idx in self.split_blocks:
            original_mlp = self.vit_base.vit.blocks[block_idx].mlp
            moe_mlp = SplitMoEFFN(original_mlp)
            self.vit_base.vit.blocks[block_idx].mlp = moe_mlp
            self.moe_blocks[block_idx] = moe_mlp

        # 5. Dual Classifier Heads (initialized from pretrained head)
        original_head = self.vit_base.vit.head
        self.head_low = copy.deepcopy(original_head)
        self.head_high = copy.deepcopy(original_head)

    def forward(
        self,
        waveform: Tensor,
        return_dict: bool = False,
    ) -> dict[str, Tensor] | Tensor:
        # Standardize waveform
        wave_norm = rms_normalize(waveform)

        # Step 1: Extract 6 acoustic morphology features from waveform
        spectral_feat = self.acoustic_extractor(wave_norm)

        # Step 2: Compute Log-Mel Spectrogram
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())
        mel_aug = self.spec_augment(mel)

        # Step 3: Run patch embedding & early transformer blocks (0 to 9)
        vit = self.vit_base.vit
        x = vit.patch_embed(mel_aug[..., : self.vit_base.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))

        self.vit_base.shared.value = None

        # Execute Blocks 0 through 9
        for i in range(10):
            x = vit.blocks[i](x)

        # Extract Block 9 token representation for router
        # Mean pooling over tokens as acoustic context
        block9_token = x[:, 0]  # CLS token [B, 192]

        # Step 4: Predict dynamic soft routing weight alpha in [0, 1]
        alpha, gates = self.router(spectral_feat, block9_token)

        # Step 5: Execute deep MoE Blocks 10 & 11
        for i in range(10, 12):
            block = vit.blocks[i]
            # Attention
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            # MoE Split-FFN
            moe_mlp = block.mlp
            normed_x = block.norm2(x)
            mlp_out = moe_mlp(normed_x, alpha)
            x = x + block.drop_path2(block.ls2(mlp_out))

        # Step 6: Final LayerNorm and Dual Head Aggregation
        x = vit.norm(x)
        cls_final = x[:, 0]  # [B, 192]

        logits_low = self.head_low(cls_final)
        logits_high = self.head_high(cls_final)

        # Soft continuous fusion of expert logits:
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
