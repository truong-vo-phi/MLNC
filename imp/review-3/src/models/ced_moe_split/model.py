"""
CED-SingleMoE-SplitFFN Architecture (~6.62M Parameters).

Proposed Single-Backbone MoE Model for Robust Audio Classification under Multi-level SNR Noise:
1. Waveform n_hat (16kHz, 4s = 64,000 samples) -> RMS Normalization (target_rms = 0.07925).
2. Parallel Frontends:
   - AcousticFeatureExtractor: STFT (257 bins x 401 frames) -> 6 physics features f_spec in R^6.
   - LogMel Frontend: 64 mel bins x 398 frames -> instance z-score norm -> SpecAugment.
3. Patch Embedding & Blocks 0-9:
   - Conv2d 16x16, stride 16 -> 4x24 = 96 patch tokens + CLS -> 97x192.
   - Blocks 0..9: Shared Pre-LN Transformer blocks with 2D GREP BiasAttention.
4. Self-Contained Spectral-Token Router:
   - Intermediate CLS token h_9 [B, 192] from Block 9 -> Linear(192->16).
   - Concatenated with f_spec [B, 6] -> [B, 22] -> MLP 22->32->2 -> Softmax -> alpha in (0, 1).
5. SplitMoEFFN in Blocks 10 & 11:
   - FFN_shared (0.50) + FFN_low (0.50 * (1 - alpha)) + FFN_high (0.50 * alpha).
   - Identity warm-start at init: copies base MLP 3 times.
6. Dual Classifier Head:
   - CLS token h_final [B, 192] after Block 11 + LayerNorm.
   - Head_low: Linear(192, 36), Head_high: Linear(192, 36).
   - Fused logits: z = (1 - alpha) * z_low + alpha * z_high.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseAudioClassifier
from src.models.ced_moe_split.acoustic_frontend import (
    AcousticFeatureExtractor,
    LogMel,
    SpecAugment,
    rms_normalize,
)
from src.models.ced_moe_split.attention import BiasAttention, SharedBias
from src.models.ced_moe_split.router import SelfContainedSpectralTokenRouter
from src.models.ced_moe_split.split_ffn import SplitMoEFFN

CED_PREFIX = "encoder."


class CEDSingleMoESplitFFN(BaseAudioClassifier):
    """
    CED-SingleMoE-SplitFFN Audio Classifier (~6.62M Parameters).
    """
    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 64,
        num_frames: int = 398,
        freq_bins: int | None = None,
        frames: int | None = None,
        split_blocks: tuple[int, ...] = (10, 11),
        dropout: float = 0.0,
        load_pretrained: bool = True,
        pretrained_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            sample_rate=sample_rate,
            n_mels=n_mels if freq_bins is None else freq_bins,
            num_frames=num_frames if frames is None else frames,
        )
        self.split_blocks = split_blocks
        self.num_classes = num_classes
        self.freq_bins = n_mels if freq_bins is None else freq_bins
        self.frames = num_frames if frames is None else frames
        self.freq_patches = self.freq_bins // 16  # 4
        self.time_patches = self.frames // 16     # 24

        # 1. Frontend processors
        self.frontend = LogMel(
            sample_rate=sample_rate,
            n_mels=self.freq_bins,
            num_frames=self.frames,
        )
        self.spec_augment = SpecAugment()
        self.acoustic_extractor = AcousticFeatureExtractor()

        # 2. Base ViT backbone (deit_tiny_patch16_224)
        vit = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes,
            img_size=(self.freq_bins, self.frames),
            drop_rate=dropout,
            in_chans=1,
        )

        # Patch embedding adjustment for spectrogram [1, 64, 384]
        vit.patch_embed.proj = nn.Conv2d(1, 192, kernel_size=(16, 16), stride=(16, 16))
        n_patches = self.freq_patches * self.time_patches  # 4 * 24 = 96
        vit.patch_embed.num_patches = n_patches
        vit.patch_embed.img_size = (self.freq_bins, self.time_patches * 16)
        vit.patch_embed.grid_size = (self.freq_patches, self.time_patches)
        vit.patch_embed.dynamic_img_pad = False

        # 2D Positional embedding: [1, 97, 192] (1 CLS + 96 patches)
        vit.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, 192))
        nn.init.trunc_normal_(vit.pos_embed, std=0.02)

        # Equip each block with GREP BiasAttention
        self.shared_bias = SharedBias()
        for idx, block in enumerate(vit.blocks):
            owns_table = (idx == 0)
            block.attn = BiasAttention(
                base=block.attn,
                shared=self.shared_bias,
                owns_table=owns_table,
                freq_patches=self.freq_patches,
                time_patches=self.time_patches,
                has_cls=True,
            )

        self.vit = vit

        # 3. Router
        self.router = SelfContainedSpectralTokenRouter(
            spectral_dim=6,
            token_dim=192,
            token_proj_dim=16,
            hidden_dim=32,
        )

        # 4. Replace MLPs in split_blocks (Blocks 10 & 11) with SplitMoEFFN
        self.moe_blocks: dict[int, SplitMoEFFN] = {}
        for block_idx in self.split_blocks:
            original_mlp = self.vit.blocks[block_idx].mlp
            moe_mlp = SplitMoEFFN(original_mlp)
            self.vit.blocks[block_idx].mlp = moe_mlp
            self.moe_blocks[block_idx] = moe_mlp

        # 5. Dual classifier heads (cloned from original head)
        self.head_low = copy.deepcopy(self.vit.head)
        self.head_high = copy.deepcopy(self.vit.head)

        # 6. Load pretrained weights if available
        if load_pretrained:
            self._load_ced_tiny_weights(pretrained_path)

    def forward_features(self, x: Tensor, alpha: Tensor | None = None) -> Tensor:
        """Forward through backbone given spectrogram [B, 1, 64, 398]."""
        vit = self.vit
        target_frames = self.time_patches * 16  # 24 * 16 = 384
        x_tok = vit.patch_embed(x[..., :target_frames])
        x_tok = x_tok + vit.pos_embed[:, 1:]
        B, N, C = x_tok.shape
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x_tok = vit.norm_pre(vit.pos_drop(torch.cat([cls, x_tok], dim=1)))  # [B, 97, 192]

        self.shared_bias.value = None

        # Blocks 0 to 9
        for i in range(10):
            x_tok = vit.blocks[i](x_tok)

        block9_token = x_tok[:, 0]

        if alpha is None:
            # If no physical spectral feature provided, route using zero placeholder or token alone
            dummy_spec = torch.zeros(B, 6, device=x.device, dtype=x.dtype)
            alpha, _ = self.router(dummy_spec, block9_token)

        # Blocks 10 & 11 (SplitMoEFFN)
        for i in self.split_blocks:
            block = vit.blocks[i]
            x_tok = x_tok + block.drop_path1(block.ls1(block.attn(block.norm1(x_tok))))
            moe_mlp = block.mlp
            normed_x = block.norm2(x_tok)
            mlp_out = moe_mlp(normed_x, alpha)
            x_tok = x_tok + block.drop_path2(block.ls2(mlp_out))

        return vit.norm(x_tok)

    def forward_head(self, feat: Tensor) -> Tensor:
        """Forward CLS token through fused head."""
        cls_feat = feat[:, 0]
        # Average heads when no alpha is provided
        return 0.5 * self.head_low(cls_feat) + 0.5 * self.head_high(cls_feat)

    def _load_ced_tiny_weights(self, path: str | None = None) -> None:
        """Loads offline CED-Tiny weights if available."""
        candidate_paths = [
            path,
            "pretrained/model.safetensors",
            "../pretrained/model.safetensors",
            "../../pretrained/model.safetensors",
            os.path.join(os.path.dirname(__file__), "../../../../../pretrained/model.safetensors"),
        ]
        target_file = None
        for p in candidate_paths:
            if p and os.path.isfile(p):
                target_file = p
                break

        if not target_file:
            return

        try:
            from safetensors.torch import load_file
            state = load_file(target_file)
            mapped = {k[len(CED_PREFIX):]: v for k, v in state.items() if k.startswith(CED_PREFIX + "blocks.")}
            mapped.update({
                k[len(CED_PREFIX):]: v
                for k, v in state.items()
                if k.startswith((CED_PREFIX + "patch_embed.", CED_PREFIX + "norm."))
            })
            own = self.vit.state_dict()
            mapped = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
            self.vit.load_state_dict(mapped, strict=False)

            # Re-sync SplitMoEFFN experts with loaded base MLPs
            for block_idx in self.split_blocks:
                loaded_mlp = self.vit.blocks[block_idx].mlp.ffn_shared
                self.vit.blocks[block_idx].mlp.ffn_low = copy.deepcopy(loaded_mlp)
                self.vit.blocks[block_idx].mlp.ffn_high = copy.deepcopy(loaded_mlp)

            # Re-sync dual heads
            self.head_low = copy.deepcopy(self.vit.head)
            self.head_high = copy.deepcopy(self.vit.head)
        except Exception:
            pass

    def extract_features(self, x: Tensor) -> Tensor:
        """Extracts CLS token representation [B, 192]."""
        dict_out = self.forward_full(x)
        return dict_out["cls_final"]

    def forward_full(self, waveform: Tensor) -> dict[str, Tensor]:
        """Complete forward pass returning all intermediate tokens and routing gates."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # 1. RMS Normalization
        wave_norm = rms_normalize(waveform)

        # 2. 6 Physics Features
        spectral_feat = self.acoustic_extractor(wave_norm)  # [B, 6]

        # 3. Log-Mel spectrogram + SpecAugment
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())  # [B, 1, 64, 398]
        mel_aug = self.spec_augment(mel)

        # 4. Patch Embedding & Pre-LN
        vit = self.vit
        target_frames = self.time_patches * 16  # 24 * 16 = 384
        x = vit.patch_embed(mel_aug[..., :target_frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))  # [B, 97, 192]

        self.shared_bias.value = None

        # 5. Shared Transformer blocks 0 to 9
        for i in range(10):
            x = vit.blocks[i](x)

        block9_token = x[:, 0]  # [B, 192]

        # 6. Router predicts alpha in (0, 1)
        alpha, gates = self.router(spectral_feat, block9_token)

        # 7. MoE blocks 10 & 11
        for i in self.split_blocks:
            block = vit.blocks[i]
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            moe_mlp = block.mlp
            normed_x = block.norm2(x)
            mlp_out = moe_mlp(normed_x, alpha)
            x = x + block.drop_path2(block.ls2(mlp_out))

        # 8. LayerNorm and Dual Heads
        x = vit.norm(x)
        cls_final = x[:, 0]  # [B, 192]

        logits_low = self.head_low(cls_final)
        logits_high = self.head_high(cls_final)

        logits_fused = (1.0 - alpha) * logits_low + alpha * logits_high

        return {
            "logits": logits_fused,
            "logits_low": logits_low,
            "logits_high": logits_high,
            "alpha": alpha,
            "gates": gates,
            "spectral_feat": spectral_feat,
            "block9_token": block9_token,
            "cls_final": cls_final,
        }

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | dict[str, Tensor]:
        """
        Unified forward pass compatible with BaseAudioClassifier:
        - By default returns logits [B, 36].
        - If return_features=True, returns (logits, cls_final).
        - If return_dict=True, returns full dictionary.
        """
        if x.ndim == 4:
            # Spectrogram input [B, 1, F, T]
            feat = self.forward_features(x)
            logits = self.forward_head(feat)
            if return_features:
                return logits, feat[:, 0]
            return logits

        dict_out = self.forward_full(x)
        if return_dict:
            return dict_out
        if return_features:
            return dict_out["logits"], dict_out["cls_final"]
        return dict_out["logits"]

    def get_num_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        router = sum(p.numel() for p in self.router.parameters())
        return {"total": total, "trainable": trainable, "router": router}
