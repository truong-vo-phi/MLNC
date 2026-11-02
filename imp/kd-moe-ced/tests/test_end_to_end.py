"""
Kiểm thử Tích hợp Đầu-Cuối (End-to-End Integration Tests) cho Pipeline CEDSingleMoESplitFFN.

Xác minh:
1. Lượt truyền thẳng hoàn chỉnh từ sóng âm thô [B, 64000] tới logits 36 lớp [B, 36].
2. Cơ chế Zero-Oracle: suy luận hoàn toàn tự thân từ sóng âm không cần nhãn SNR.
3. Luồng gradient truyền ngược đầy đủ qua toàn bộ đồ thị tính toán.
"""

import pytest
import torch
from src.models.split_moe import CEDSingleMoESplitFFN


def test_end_to_end_forward_pass(sample_waveform):
    """Kiểm tra lượt truyền thẳng toàn bộ mô hình và kiểm chứng kích thước các tensor đầu ra."""
    model = CEDSingleMoESplitFFN(num_classes=36, freq_bins=64, frames=398)
    model.eval()

    with torch.no_grad():
        out = model(sample_waveform, return_dict=True)

    # 1. Các trường dữ liệu trong dict đầu ra
    assert "logits" in out
    assert "logits_low" in out
    assert "logits_high" in out
    assert "alpha" in out
    assert "gates" in out
    assert "spectral_feat" in out
    assert "block9_token" in out

    # 2. Kích thước các tensor
    assert out["logits"].shape == (2, 36)
    assert out["logits_low"].shape == (2, 36)
    assert out["logits_high"].shape == (2, 36)
    assert out["alpha"].shape == (2, 1)
    assert out["gates"].shape == (2, 2)
    assert out["spectral_feat"].shape == (2, 6)
    assert out["block9_token"].shape == (2, 192)

    # 3. Tính hợp lệ về mặt số học
    assert not torch.isnan(out["logits"]).any()
    assert not torch.isinf(out["logits"]).any()
    assert (out["alpha"] >= 0.0).all() and (out["alpha"] <= 1.0).all()


def test_full_backward_gradient_flow(sample_waveform):
    """Kiểm tra luồng lan truyền gradient ngược qua từng thành phần của mạng."""
    model = CEDSingleMoESplitFFN(num_classes=36, freq_bins=64, frames=398)
    model.train()

    out = model(sample_waveform, return_dict=True)
    loss = out["logits"].sum() + out["alpha"].sum()
    loss.backward()

    # 1. Gradient trong Backbone sớm (Patch Embedding & Block 0)
    assert model.vit_base.vit.patch_embed.proj.weight.grad is not None
    assert model.vit_base.vit.blocks[0].mlp.fc1.weight.grad is not None

    # 2. Gradient trong Bộ định tuyến Router
    assert model.router.token_proj[1].weight.grad is not None
    assert model.router.mlp[0].weight.grad is not None
    assert model.router.mlp[3].weight.grad is not None

    # 3. Gradient trong các Chuyên gia MoE sâu (Blocks 10 & 11)
    moe10 = model.vit_base.vit.blocks[10].mlp
    assert moe10.ffn_shared.fc1.weight.grad is not None
    assert moe10.ffn_low.fc1.weight.grad is not None
    assert moe10.ffn_high.fc1.weight.grad is not None

    # 4. Gradient trong Đầu phân loại kép
    assert model.head_low.weight.grad is not None
    assert model.head_high.weight.grad is not None
