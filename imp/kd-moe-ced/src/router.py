"""
Bộ định tuyến Phổ - Token Tự thân (Self-Contained Spectral-Token Router) cho Mô hình Single-Backbone MoE (Zero-Oracle SNR).

Module này dự đoán trọng số điều phối chuyên gia liên tục alpha trong khoảng (0, 1) một cách linh hoạt
dựa trên sự kết hợp giữa:
1. 6 đặc trưng vật lý âm học / hình thái phổ nội tại (f_spec thuộc R^6) trích xuất trực tiếp từ sóng âm.
2. Biểu diễn ngữ cảnh ngữ nghĩa trung gian của CLS token (h_9 thuộc R^192) trích xuất sau Transformer Block 9.

Tài liệu tham khảo:
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE", ACL 2024. arXiv:2401.06066.
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts", NeurIPS 2021. arXiv:2106.05974.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfContainedSpectralTokenRouter(nn.Module):
    """
    Module định tuyến động tự thân (Self-Contained) ước lượng trọng số cổng mềm liên tục alpha.

    Kiến trúc Luồng Dữ liệu (Dataflow Architecture):
        1. Token CLS h_9 [B, 192] sau Khối Transformer Block 9:
           -> LayerNorm(192) -> Linear(192, 16) -> GELU -> h_tilde_9 [B, 16]
        2. Ghép nối với 6 đặc trưng vật lý f_spec [B, 6] từ frontend:
           -> u = [f_spec || h_tilde_9] [B, 22]
        3. Mạng MLP tương tác liên mức đặc trưng:
           -> Linear(22, 32) -> LayerNorm(32) -> GELU -> Linear(32, 2)
           -> Logits [r_low, r_high] [B, 2]
        4. Chuẩn hóa hàm Softmax:
           -> alpha = exp(r_high) / (exp(r_low) + exp(r_high)) thuộc khoảng (0, 1)

    Đầu ra:
        - alpha: [B, 1] trọng số mềm điều phối cho Chuyên gia High-SNR (và 1 - alpha cho Chuyên gia Low-SNR).
        - gates: [B, 2] phân phối xác suất chuẩn hóa trên [Low, High].
    """
    def __init__(
        self,
        spectral_dim: int = 6,
        token_dim: int = 192,
        token_proj_dim: int = 16,
        hidden_dim: int = 32,
        transition_center_db: float = 15.0,
        transition_tau: float = 2.5,
    ) -> None:
        super().__init__()
        self.transition_center_db = transition_center_db
        self.transition_tau = transition_tau

        # Nhánh 1: Tầng chiếu giảm chiều token CLS (192 -> 16)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )

        # Nhánh 2: Mạng MLP tương tác liên mức (22 -> 32 -> 2)
        in_total = spectral_dim + token_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_total, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [r_low, r_high]
        )

    def forward(
        self,
        spectral_feat: Tensor,
        intermediate_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Tham số:
            spectral_feat: [B, 6] trích xuất từ AcousticFeatureExtractor
            intermediate_token: [B, 192] biểu diễn CLS token trích xuất sau Block 9
        Trả về:
            alpha: [B, 1] trọng số mềm điều phối cho Chuyên gia High-SNR
            gates: [B, 2] xác suất softmax trên hai chuyên gia [p_low, p_high]
        """
        # Nén biểu diễn token 192 chiều xuống 16 chiều
        token_h = self.token_proj(intermediate_token)  # [B, 16]

        # Ghép nối đặc trưng vật lý trực tiếp với ngữ cảnh ngữ nghĩa trung gian
        fused_feat = torch.cat([spectral_feat, token_h], dim=-1)  # [B, 22]

        # Tính toán logits định tuyến qua MLP
        logits = self.mlp(fused_feat)  # [B, 2]

        # Phân phối xác suất qua hàm Softmax
        gates = F.softmax(logits, dim=-1)  # [B, 2]
        alpha = gates[:, 1:2]              # [B, 1] Trọng số của chuyên gia High-SNR
        return alpha, gates

    def compute_target_soft_label(self, snr_db: Tensor) -> Tensor:
        """
        Tính toán nhãn mục tiêu mềm liên tục từ giá trị SNR ground-truth trong quá trình huấn luyện:
        y_target = Sigmoid((SNR - 15.0) / 2.5)

        Bảng giá trị tham chiếu:
            SNR = -5 dB -> y_target = 0.0003  (Hoàn toàn ở chế độ Low-SNR)
            SNR =  0 dB -> y_target = 0.0025
            SNR =  5 dB -> y_target = 0.0180
            SNR = 10 dB -> y_target = 0.1192
            SNR = 15 dB -> y_target = 0.5000  (Vùng chuyển tiếp mượt mà)
            SNR = 20 dB -> y_target = 0.8808  (Chế độ High-SNR xử lý rò rỉ tiếng nói)
        """
        z = (snr_db.float() - self.transition_center_db) / self.transition_tau
        return torch.sigmoid(z).unsqueeze(-1)

    def compute_router_loss(self, alpha: Tensor, snr_db: Tensor) -> Tensor:
        """
        Hàm mất mát Binary Cross Entropy có giám sát dành riêng cho trọng số router trong huấn luyện.
        """
        target = self.compute_target_soft_label(snr_db)
        return F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), target.float())
