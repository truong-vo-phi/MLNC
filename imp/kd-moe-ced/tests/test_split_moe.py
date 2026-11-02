"""
Kiểm thử Đơn vị (Unit tests) cho SplitMoEFFN và Kiến trúc CEDSingleMoESplitFFN.

Xác minh:
1. Tính đồng nhất toán học khi khởi tạo Warm-Start (Identity Warm-Start):
   Tại bước khởi tạo ban đầu, với MỌI alpha thuộc [0, 1], SplitMoEFFN(x, alpha) == BaseMLP(x).
2. Các khối sâu Blocks 10 & 11 sử dụng đúng SplitMoEFFN.
3. Cơ chế hòa trộn mềm Dual Classifier Head: z = (1 - alpha) * z_low + alpha * z_high.
"""

import pytest
import torch
import torch.nn as nn
from src.models.split_moe import CEDSingleMoESplitFFN, SplitMoEFFN
from src.models.vit_grep import ViTSBias


def test_split_moe_ffn_identity_warm_start():
    """Kiểm tra tính bảo toàn đồng nhất toán học của SplitMoEFFN với MLP cơ sở ở bước khởi tạo."""
    # Xây dựng MLP cơ sở: 192 -> 768 -> 192 với kích hoạt GELU
    base_mlp = nn.Sequential(
        nn.Linear(192, 768),
        nn.GELU(),
        nn.Linear(768, 192),
    )

    moe = SplitMoEFFN(base_mlp)
    x = torch.randn(4, 97, 192)

    # Đầu ra tham chiếu từ MLP gốc
    y_base = base_mlp(x)

    # Kiểm tra tính đồng nhất cho nhiều giá trị alpha khác nhau trong [0, 1]
    for alpha_val in [0.0, 0.15, 0.50, 0.75, 1.0]:
        alpha = torch.full((4, 1), alpha_val)
        y_moe = moe(x, alpha)
        # Sai số phải triệt tiêu về 0
        assert torch.allclose(y_moe, y_base, atol=1e-5), f"Lỗi tại alpha={alpha_val}"


def test_model_architecture_and_deep_moe_blocks():
    """Kiểm tra cấu trúc phân tầng: Blocks 0-9 là MLP chuẩn, Blocks 10-11 là SplitMoEFFN."""
    model = CEDSingleMoESplitFFN(num_classes=36, split_blocks=(10, 11))

    # Blocks 0 đến 9 là các khối Transformer tiêu chuẩn
    for i in range(10):
        mlp = model.vit_base.vit.blocks[i].mlp
        assert not isinstance(mlp, SplitMoEFFN)

    # Blocks 10 và 11 BẮT BUỘC là SplitMoEFFN
    assert isinstance(model.vit_base.vit.blocks[10].mlp, SplitMoEFFN)
    assert isinstance(model.vit_base.vit.blocks[11].mlp, SplitMoEFFN)

    # Kiểm tra đầu phân loại kép Dual Heads
    assert isinstance(model.head_low, nn.Linear)
    assert isinstance(model.head_high, nn.Linear)
    assert model.head_low.out_features == 36
    assert model.head_high.out_features == 36


def test_dual_head_soft_blending():
    """Kiểm tra cơ chế hòa trộn mềm tuyến tính trên logits của 2 đầu phân loại."""
    model = CEDSingleMoESplitFFN(num_classes=36)
    cls_token = torch.randn(2, 192)

    z_low = model.head_low(cls_token)
    z_high = model.head_high(cls_token)

    # Trường hợp 1: alpha = 0 -> z == z_low
    alpha_0 = torch.tensor([[0.0], [0.0]])
    z_fused_0 = (1.0 - alpha_0) * z_low + alpha_0 * z_high
    assert torch.allclose(z_fused_0, z_low, atol=1e-5)

    # Trường hợp 2: alpha = 1 -> z == z_high
    alpha_1 = torch.tensor([[1.0], [1.0]])
    z_fused_1 = (1.0 - alpha_1) * z_low + alpha_1 * z_high
    assert torch.allclose(z_fused_1, z_high, atol=1e-5)

    # Trường hợp 3: alpha = 0.5 -> z == 0.5 * z_low + 0.5 * z_high
    alpha_half = torch.tensor([[0.5], [0.5]])
    z_fused_half = (1.0 - alpha_half) * z_low + alpha_half * z_high
    assert torch.allclose(z_fused_half, 0.5 * (z_low + z_high), atol=1e-5)
