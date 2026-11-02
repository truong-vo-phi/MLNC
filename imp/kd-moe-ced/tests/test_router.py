"""
Kiểm thử Đơn vị (Unit tests) cho Bộ định tuyến Phổ - Token Tự thân (Zero-Oracle SNR).

Xác minh:
1. Nhánh chiếu token nén CLS token từ 192 chiều xuống 16 chiều.
2. Ghép nối f_spec (6 chiều) và token_h (16 chiều) tạo vector 22 chiều.
3. Tầng MLP trả về 2 logits [r_low, r_high].
4. Softmax sinh ra trọng số mềm liên tục alpha thuộc khoảng (0, 1).
5. Hàm nhãn mục tiêu mềm tăng đơn điệu theo SNR.
6. Hành vi tính toán của hàm mất mát Router BCE loss.
"""

import torch
from src.router import SelfContainedSpectralTokenRouter


def test_router_forward_shapes_and_values():
    """Kiểm tra kích thước đầu ra và tính chất cổng mềm của Router."""
    router = SelfContainedSpectralTokenRouter(
        spectral_dim=6,
        token_dim=192,
        token_proj_dim=16,
        hidden_dim=32,
    )

    B = 4
    spectral_feat = torch.randn(B, 6)
    intermediate_token = torch.randn(B, 192)

    alpha, gates = router(spectral_feat, intermediate_token)

    # Kiểm tra kích thước tensor đầu ra
    assert alpha.shape == (B, 1)
    assert gates.shape == (B, 2)

    # Tính chất của cổng mềm:
    # 1. Khoảng giá trị liên tục: alpha thuộc (0, 1)
    assert (alpha > 0.0).all() and (alpha < 1.0).all()

    # 2. Tổng xác suất bằng 1 trên 2 cổng
    assert torch.allclose(gates.sum(dim=-1), torch.ones(B), atol=1e-5)

    # 3. alpha bằng đúng gates[:, 1:2]
    assert torch.allclose(alpha, gates[:, 1:2])


def test_target_soft_label_monotonicity():
    """Kiểm tra tính đơn điệu tăng ngặt của nhãn mục tiêu mềm theo mức SNR."""
    router = SelfContainedSpectralTokenRouter()

    snrs = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    targets = router.compute_target_soft_label(snrs).squeeze(-1)

    # Kiểm tra tính đơn điệu
    for i in range(len(targets) - 1):
        assert targets[i] < targets[i + 1]

    # Các mốc giá trị tham chiếu:
    # SNR = 15 dB là điểm trung tâm chuyển tiếp -> target == 0.5
    assert torch.isclose(targets[4], torch.tensor(0.5), atol=1e-3)
    # SNR = -5 dB tiến rất sát 0
    assert targets[0] < 0.01
    # SNR = 20 dB rất cao (> 0.8)
    assert targets[5] > 0.85


def test_router_bce_loss():
    """Kiểm tra hàm mất mát BCE của router có giá trị không âm và không chứa NaN."""
    router = SelfContainedSpectralTokenRouter()
    alpha = torch.tensor([[0.1], [0.85]])
    snr = torch.tensor([-5.0, 20.0])

    loss = router.compute_router_loss(alpha, snr)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
