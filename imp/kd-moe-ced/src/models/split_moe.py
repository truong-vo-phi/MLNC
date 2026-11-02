"""
Kiến trúc Single-Backbone CED-Tiny kết hợp Split-FFN MoE và Chuyên gia Dùng chung (CEDSingleMoESplitFFN).

Luồng Kiến trúc Xuyên suốt:
    1. Sóng âm n_hat (16kHz, 4.0 giây = 64,000 mẫu)
       -> Chuẩn hóa Năng lượng RMS (target_rms = 0.07925)
    2. Hai Nhánh Frontend Song Song:
       -> AcousticFeatureExtractor: STFT (257 bins x 401 frames) -> 6 đặc trưng vật lý f_spec thuộc R^6
       -> LogMel Frontend: (64 dải Mel x 398 frames) -> Chuẩn hóa mẫu (Instance Norm) -> SpecAugment
    3. Tầng Nhúng Patch & Các Khối Transformer Đầu (Blocks 0..9):
       -> Conv2d(1, 192, 16x16, stride 16) -> 4x24 = 96 patch tokens
       -> Ghép CLS token + 2D Positional Embeddings -> X_0 [B, 97, 192]
       -> Blocks 0 đến 9 (Khối dùng chung hoàn toàn): Cấu trúc Pre-LN với 2D GREP BiasAttention
    4. Bộ Định Tuyến Tự Thân (Self-Contained Spectral-Token Router):
       -> Trích xuất CLS token h_9 [B, 192] từ Block 9
       -> Nén 192 -> 16 + Ghép với f_spec [B, 6] -> R^22 -> MLP 22 -> 32 -> 2 -> Softmax -> alpha thuộc (0, 1)
    5. Các Khối MoE Sâu (Blocks 10 & 11) dạng SplitMoEFFN:
       -> FFN_shared (trọng số 0.5) + FFN_low (0.5 * (1 - alpha)) + FFN_high (0.5 * alpha)
       -> Khởi tạo Warm-start bảo toàn (hoàn toàn tương đương CED-Tiny gốc tại bước 0)
    6. Đầu Phân Loại Kép (Dual Classifier Head):
       -> CLS token h_final [B, 192] sau Block 11 + LayerNorm
       -> Head_low: Linear(192, 36)
       -> Head_high: Linear(192, 36)
       -> Logits cuối cùng: z = (1 - alpha) * z_low + alpha * z_high

Tài liệu tham khảo:
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts", NeurIPS 2021. arXiv:2106.05974.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE", ACL 2024. arXiv:2401.06066.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.frontend import AcousticFeatureExtractor, LogMel, SpecAugment, rms_normalize
from src.models.vit_grep import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
from src.router import SelfContainedSpectralTokenRouter


class SplitMoEFFN(nn.Module):
    """
    Cấu trúc Split-FFN gồm 1 Chuyên gia Dùng chung (Shared Expert) + 2 Chuyên gia Định tuyến (Low-SNR & High-SNR).

    Công thức Toán học:
        y = 0.5 * FFN_shared(x) + 0.5 * [(1 - alpha) * FFN_low(x) + alpha * FFN_high(x)]

    Tổng trọng số điều phối:
        0.5 + 0.5 * (1 - alpha + alpha) = 1.0 (Tổ hợp lồi chuẩn hóa tuyệt đối).

    Khởi tạo Identity Warm-Start:
        Khi ffn_shared, ffn_low, và ffn_high được sao chép nguyên bản từ base_mlp tiền huấn luyện,
        tại step 0:
            y = 0.5 * M(x) + 0.5 * [(1 - alpha) * M(x) + alpha * M(x)]
              = 0.5 * M(x) + 0.5 * M(x) = M(x)
        Đầu ra HOÀN TOÀN ĐỒNG NHẤT VỀ MẶT TOÁN HỌC với CED-Tiny chuẩn cho MỌI giá trị alpha thuộc (0, 1).
    """
    def __init__(self, base_mlp: nn.Module) -> None:
        super().__init__()
        self.ffn_shared = copy.deepcopy(base_mlp)
        self.ffn_low = copy.deepcopy(base_mlp)
        self.ffn_high = copy.deepcopy(base_mlp)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        """
        Tham số:
            x: [B, Tokens, C] (ví dụ: [B, 97, 192])
            alpha: [B, 1] hoặc [B, 1, 1] trọng số điều phối mềm
        Trả về:
            out: [B, Tokens, C]
        """
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]

        out_shared = self.ffn_shared(x)
        out_low = self.ffn_low(x)
        out_high = self.ffn_high(x)

        # Tổ hợp lồi chuẩn hóa
        routed_out = (1.0 - alpha) * out_low + alpha * out_high
        return 0.5 * out_shared + 0.5 * routed_out


class CEDSingleMoESplitFFN(nn.Module):
    """
    Kiến trúc toàn vẹn Single-Backbone CED-Tiny MoE (~6.62M tham số).

    - Blocks 0 đến 9: Các khối Transformer dùng chung 100% với cơ chế 2D GREP BiasAttention.
    - Blocks 10 & 11: Khối Split-FFN chứa 1 Chuyên gia dùng chung + 2 Chuyên gia chuyên biệt (Low & High).
    - Bộ phân loại: Dual-Head (Head_low và Head_high) hòa trộn mềm qua trọng số alpha.
    - Bộ định tuyến Tự thân (Self-Contained Router): Sử dụng 6 đặc trưng vật lý từ AcousticFeatureExtractor
      kết hợp với biểu diễn trung gian của CLS token từ Block 9.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        split_blocks: tuple[int, ...] = (10, 11),
        dropout: float = 0.0,
        load_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.split_blocks = split_blocks
        self.num_classes = num_classes

        # 1. Các bộ xử lý âm thanh Frontend
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.acoustic_extractor = AcousticFeatureExtractor()

        # 2. Backbone cơ sở (Pre-LN ViT với 2D GREP)
        self.vit_base = ViTSBias(
            num_classes=num_classes,
            freq_bins=freq_bins,
            frames=frames,
            pretrained=False,
            dropout=dropout,
            use_bias=True,
            time_major=False,
        )
        if load_pretrained:
            load_ced_tiny(self.vit_base)

        # 3. Bộ định tuyến Tự thân (Spectral-Token Router)
        self.router = SelfContainedSpectralTokenRouter(
            spectral_dim=6,
            token_dim=192,
            token_proj_dim=16,
            hidden_dim=32,
        )

        # 4. Thay thế tầng MLP trong các khối sâu (10 & 11) bằng SplitMoEFFN
        self.moe_blocks: dict[int, SplitMoEFFN] = {}
        for block_idx in self.split_blocks:
            original_mlp = self.vit_base.vit.blocks[block_idx].mlp
            moe_mlp = SplitMoEFFN(original_mlp)
            self.vit_base.vit.blocks[block_idx].mlp = moe_mlp
            self.moe_blocks[block_idx] = moe_mlp

        # 5. Đầu phân loại kép Dual-Head (khởi tạo từ head gốc tiền huấn luyện)
        original_head = self.vit_base.vit.head
        self.head_low = copy.deepcopy(original_head)
        self.head_high = copy.deepcopy(original_head)

    def forward(
        self,
        waveform: Tensor,
        return_dict: bool = False,
    ) -> dict[str, Tensor] | Tensor:
        """
        Lượt truyền thẳng End-to-End:
            Sóng âm -> Đặc trưng vật lý & Phổ Mel -> Blocks 0..9 -> Router alpha -> Blocks 10..11 -> Đầu phân loại kép
        """
        # Bước 1: Chuẩn hóa năng lượng RMS (target_rms = 0.07925)
        wave_norm = rms_normalize(waveform)

        # Bước 2: Trích xuất 6 đặc trưng vật lý âm học từ sóng âm đã chuẩn hóa
        spectral_feat = self.acoustic_extractor(wave_norm)  # [B, 6]

        # Bước 3: Tính toán phổ Log-Mel và áp dụng SpecAugment
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())  # [B, 1, 64, 398]
        mel_aug = self.spec_augment(mel)

        # Bước 4: Chiếu Patch Embedding và chuẩn hóa trước (Pre-LN)
        vit = self.vit_base.vit
        # Conv2d 16x16, stride 16 -> 4 x 24 = 96 patch tokens
        x = vit.patch_embed(mel_aug[..., : self.vit_base.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))  # [B, 97, 192]

        self.vit_base.shared.value = None

        # Bước 5: Truyền qua các khối dùng chung từ Block 0 đến Block 9
        for i in range(10):
            x = vit.blocks[i](x)

        # Trích xuất biểu diễn token CLS tại Block 9 cho bộ định tuyến
        block9_token = x[:, 0]  # [B, 192]

        # Bước 6: Bộ định tuyến dự đoán trọng số mềm alpha thuộc khoảng (0, 1)
        alpha, gates = self.router(spectral_feat, block9_token)

        # Bước 7: Truyền qua các khối MoE sâu (Blocks 10 & 11)
        for i in self.split_blocks:
            block = vit.blocks[i]
            # Attention với cơ chế 2D GREP
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            # Tầng SplitMoEFFN điều chế theo alpha
            moe_mlp = block.mlp
            normed_x = block.norm2(x)
            mlp_out = moe_mlp(normed_x, alpha)
            x = x + block.drop_path2(block.ls2(mlp_out))

        # Bước 8: Chuẩn hóa LayerNorm cuối và tổng hợp từ Dual Heads
        x = vit.norm(x)
        cls_final = x[:, 0]  # [B, 192]

        logits_low = self.head_low(cls_final)   # [B, 36]
        logits_high = self.head_high(cls_final) # [B, 36]

        # Kết hợp mềm liên tục: z = (1 - alpha) * z_low + alpha * z_high
        logits_fused = (1.0 - alpha) * logits_low + alpha * logits_high

        if return_dict:
            return {
                "logits": logits_fused,
                "logits_low": logits_low,
                "logits_high": logits_high,
                "alpha": alpha,
                "gates": gates,
                "spectral_feat": spectral_feat,
                "block9_token": block9_token,
            }
        return logits_fused

    def get_num_params(self) -> dict[str, int]:
        """Thống kê tổng số lượng tham số của mô hình."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        router = sum(p.numel() for p in self.router.parameters())
        return {"total": total, "trainable": trainable, "router": router}
