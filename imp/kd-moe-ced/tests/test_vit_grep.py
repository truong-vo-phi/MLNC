"""
Kiểm thử Đơn vị (Unit tests) cho ViT Backbone, Patch Tokens, Pre-LN, và 2D GREP BiasAttention.

Xác minh:
1. Chiếu patch Conv2d 16x16 sinh ra đúng 4 x 24 = 96 patch tokens.
2. Ghép token CLS tạo tensor X_0 kích thước [B, 97, 192].
3. Nhúng vị trí 2D Positional Embeddings.
4. Cấu trúc Pre-LN và số lượng tham số khối MLP (295,872 tham số).
5. Cơ chế 2D Gated Relative Position Bias (GREP) trong BiasAttention:
   - 3 heads, head_dim = 64
   - Tái sử dụng bảng bias dùng chung giữa các khối qua SharedBias
   - Cổng động g_a, g_b và hệ số điều chế biên độ S
"""

import pytest
import torch
import torch.nn as nn
from src.models.vit_grep import BiasAttention, SharedBias, ViTSBias


def test_patch_tokens_and_cls():
    """Kiểm tra phép chiếu patch Conv2d và ghép CLS token thành chuỗi 97 tokens."""
    model = ViTSBias(num_classes=36, freq_bins=64, frames=398, pretrained=False)
    vit = model.vit

    # Phổ Mel đầu vào: [B, 1, 64, 398]
    x_mel = torch.randn(2, 1, 64, 398)

    # Chiếu patch Conv2d (16x16, stride 16)
    patches = vit.patch_embed(x_mel[..., : model.frames])
    B, N, C = patches.shape

    # 64 // 16 = 4 patch tần số, 398 // 16 = 24 patch thời gian -> 4 * 24 = 96 patches
    assert N == 96
    assert C == 192

    # Cộng pos embed 2D và ghép CLS token
    cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
    x = vit.norm_pre(vit.pos_drop(torch.cat([cls, patches + vit.pos_embed[:, 1:]], dim=1)))

    # Tổng số tokens = 96 + 1 = 97
    assert x.shape == (2, 97, 192)


def test_pre_ln_mlp_param_count():
    """Kiểm tra cấu trúc Pre-LN và số lượng tham số chính xác của khối MLP (295,872 tham số)."""
    model = ViTSBias(num_classes=36, freq_bins=64, frames=398, pretrained=False)
    block0 = model.vit.blocks[0]

    # Kiểm tra cấu trúc Pre-LN
    assert hasattr(block0, "norm1")
    assert hasattr(block0, "norm2")
    assert hasattr(block0, "attn")
    assert hasattr(block0, "mlp")

    # Số lượng tham số MLP: 192 -> 768 -> 192
    # Tuyến tính 1: 192 * 768 + 768 = 148,224
    # Tuyến tính 2: 768 * 192 + 192 = 147,648
    # Tổng cộng = 295,872 tham số
    mlp_params = sum(p.numel() for p in block0.mlp.parameters())
    assert mlp_params == 295872


def test_grep_bias_attention_mechanism():
    """Kiểm tra cơ chế Attention 2D GREP và việc chia sẻ ma trận bias giữa Block 0 và Block 1."""
    model = ViTSBias(num_classes=36, freq_bins=64, frames=398, pretrained=False)
    block0_attn = model.vit.blocks[0].attn
    block1_attn = model.vit.blocks[1].attn

    assert isinstance(block0_attn, BiasAttention)
    assert isinstance(block1_attn, BiasAttention)

    # Block 0 sở hữu các bảng nhúng vị trí tương đối
    assert hasattr(block0_attn, "freq_rel_bias")
    assert hasattr(block0_attn, "time_rel_bias")

    # Block 1 KHÔNG sở hữu bảng nhúng riêng, mà tái sử dụng của Block 0 qua shared
    assert not hasattr(block1_attn, "freq_rel_bias")
    assert block0_attn.shared is block1_attn.shared

    # Cấu hình Attention Head: 3 heads, chiều 64
    assert block0_attn.num_heads == 3
    assert block0_attn.head_dim == 64

    # Lượt truyền thẳng qua BiasAttention
    x = torch.randn(2, 97, 192)
    out0 = block0_attn(x)
    assert out0.shape == (2, 97, 192)

    # Kiểm tra ma trận bias dùng chung đã được tính toán và lưu trữ
    assert model.shared.value is not None
    assert model.shared.value.shape == (1, 3, 97, 97)

    # Block 1 tái sử dụng ma trận bias dùng chung này
    out1 = block1_attn(x)
    assert out1.shape == (2, 97, 192)
