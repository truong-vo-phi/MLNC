"""
Backbone Vision Transformer với cơ chế 2D Gated Relative Position Bias (GREP Attention).

Bao gồm các thành phần:
1. Chiếu Patch (Patch Projection): Conv2d 16x16, stride 16 (192 kênh) -> 4 x 24 = 96 patch tokens.
2. 2D Positional Embedding + CLS Token -> Chuỗi token độ dài 97.
3. Các khối Transformer chuẩn hóa trước (Pre-LN Transformer Blocks):
   - Nhánh 1: LayerNorm1 -> BiasAttention -> Kết nối tắt Residual 1
   - Nhánh 2: LayerNorm2 -> MLP 192 -> 768 -> 192 (kích hoạt GELU) -> Kết nối tắt Residual 2
4. Cơ chế 2D Gated Relative Position Bias (GREP):
   - Bảng nhúng khoảng cách tương đối 2D dùng chung (Khối 0 sở hữu, các Khối 1-11 tái sử dụng qua SharedBias).
   - Cổng điều chế động (Dynamic Gating): Vector truy vấn Q đi qua Linear(64 -> 8) + Sigmoid -> g_a, g_b.
   - Điều chế biên độ: S = g_a * (g_b * a - 1) + 2.
   - Ma trận Bias điều chế: B_attn = S * P_rel cộng trực tiếp vào QK^T / sqrt(64) trước khi qua Softmax.

Tài liệu tham khảo:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
    - BEATs: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023. arXiv:2212.09058.
    - DeiT: Touvron et al., "Training data-efficient image transformers & distillation through attention", ICML 2021. arXiv:2012.12877.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GATE_HIDDEN = 8
CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


class SharedBias:
    """Vật chứa (container) dùng chung cho ma trận bias vị trí tương đối, tính toán 1 lần duy nhất mỗi lượt forward."""
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """
    Cơ chế 2D Gated Relative Position Bias Attention (GREP Attention).
    Gồm 3 attention heads, head_dim = 64 (tổng embed_dim = 192).

    Công thức Toán học:
        1. Vector Query Q, Key K, Value V: [B, 3, 97, 64]
        2. Ma trận Bias Vị trí Tương đối 2D Tĩnh Dùng Chung P_rel:
           Xây dựng từ lưới delta 2D: delta_f thuộc [-3, 3], delta_t thuộc [-23, 23].
           Khối 0 sở hữu bảng nhúng; các khối tiếp theo tái sử dụng ma trận dùng chung này.
        3. Cổng Điều Chế Động GREP:
           gate = sigmoid(Linear_64->8(Q)) -> tách thành g_a, g_b thuộc R^[B, 3, 97, 4]
           hệ số tỷ lệ S = g_a * (g_b * a - 1.0) + 2.0
        4. Ma trận Bias Đã Điều Chế:
           B_attn = S * P_rel
        5. Scaled Dot-Product Attention:
           Attention(Q, K, V) = Softmax((Q K^T) / sqrt(64) + B_attn) V
    """
    def __init__(
        self,
        base: nn.Module,
        shared: SharedBias,
        owns_table: bool,
        freq_patches: int = 4,
        time_patches: int = 24,
        has_cls: bool = True,
        bias_init: str = "zero",
    ) -> None:
        super().__init__()
        self.has_cls = has_cls
        self.num_heads = base.num_heads
        self.head_dim = base.head_dim
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.shared = shared
        self.freq_patches = freq_patches
        self.time_patches = time_patches
        n_patches = freq_patches * time_patches

        if owns_table:
            num_f = 2 * freq_patches - 1
            num_t = 2 * time_patches - 1
            self.freq_rel_bias = nn.Embedding(num_f, self.num_heads)
            self.time_rel_bias = nn.Embedding(num_t, self.num_heads)
            self.cls_to_patches = nn.Parameter(torch.zeros(1, self.num_heads, 1, n_patches))
            self.patches_to_cls = nn.Parameter(torch.zeros(1, self.num_heads, n_patches, 1))
            self.cls_to_cls = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

            if bias_init in {"normal", "random"}:
                nn.init.normal_(self.freq_rel_bias.weight, std=0.01)
                nn.init.normal_(self.time_rel_bias.weight, std=0.01)
                nn.init.normal_(self.cls_to_patches, std=0.01)
                nn.init.normal_(self.patches_to_cls, std=0.01)
                nn.init.normal_(self.cls_to_cls, std=0.01)
            else:
                self.freq_rel_bias.weight.data.zero_()
                self.time_rel_bias.weight.data.zero_()

        # Tầng tuyến tính cổng động GREP (64 -> 8)
        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        # Hệ số tỷ lệ học được
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def compute_bias(self, device: torch.device) -> Tensor:
        """
        Tính toán ma trận bias vị trí tương đối 2D tĩnh P_rel kích thước [1, 3, 97, 97].
        """
        f_p, t_p = self.freq_patches, self.time_patches
        grid_f = torch.arange(f_p, device=device).unsqueeze(1).expand(f_p, t_p).reshape(-1)
        grid_t = torch.arange(t_p, device=device).unsqueeze(0).expand(f_p, t_p).reshape(-1)
        delta_f = (grid_f.unsqueeze(1) - grid_f.unsqueeze(0)) + (f_p - 1)
        delta_t = (grid_t.unsqueeze(1) - grid_t.unsqueeze(0)) + (t_p - 1)

        bias_patch = (self.freq_rel_bias(delta_f) + self.time_rel_bias(delta_t)).permute(2, 0, 1).unsqueeze(0)
        if not self.has_cls:
            return bias_patch

        row_cls = torch.cat([self.cls_to_cls, self.cls_to_patches], dim=-1)
        rows_patch = torch.cat([self.patches_to_cls, bias_patch], dim=-1)
        return torch.cat([row_cls, rows_patch], dim=-2)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim
        # Chiếu QKV: [B, T, 3, H, d] -> [3, B, H, T, d]
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)

        if hasattr(self, "freq_rel_bias"):
            self.shared.value = self.compute_bias(x.device)
        position_bias = self.shared.value

        # Tính toán bias vị trí tương đối có điều chế cổng động (GREP):
        # q: [B, H, T, d] -> grep_linear(q): [B, H, T, 8]
        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        bias = (scale * position_bias).to(q.dtype) if position_bias is not None else None

        # Scaled Dot-Product Attention kèm ma trận bias đã điều chế
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


class ViTSBias(nn.Module):
    """
    Backbone Vision Transformer với 2D Relative Position Bias tương thích CED-Tiny:
    - 12 khối Transformer cấu trúc Pre-LN
    - Số chiều embedding: 192 (chiều MLP: 768, 295,872 tham số mỗi MLP)
    - 3 Attention Heads (head_dim: 64)
    - Lưới Patch: 4 patch tần số x 24 patch thời gian = 96 patch tokens (+1 CLS token = 97 tokens)
    - Chiếu Patch: Conv2d(1, 192, kernel_size=16, stride=16)
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        pretrained: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        arch: str = "deit_tiny_patch16_224",
        bias_init: str = "zero",
        use_bias: bool = True,
        time_major: bool = False,
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.time_major = time_major
        self.vit = timm.create_model(
            arch,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            img_size=(freq_bins, frames),
            drop_rate=dropout,
            proj_drop_rate=dropout,
            attn_drop_rate=attn_dropout,
            drop_path_rate=drop_path,
        )
        self.freq_patches = freq_bins // 16  # 64 // 16 = 4
        self.time_patches = frames // 16     # 398 // 16 = 24
        self.frames = frames
        self.shared = SharedBias()

        # Thay thế attention tiêu chuẩn bằng 2D GREP BiasAttention
        for index, block in enumerate(self.vit.blocks):
            if use_bias:
                block.attn = BiasAttention(
                    block.attn,
                    self.shared,
                    owns_table=index == 0,
                    freq_patches=self.freq_patches,
                    time_patches=self.time_patches,
                    has_cls=True,
                    bias_init=bias_init,
                )

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        # Chiếu patch Conv2d 16x16: [B, 1, 64, 384] -> [B, 192, 4, 24] -> [B, 96, 192]
        x = vit.patch_embed(x[..., : self.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        if self.time_major:
            x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        # Ghép token CLS vào đầu chuỗi -> [B, 97, 192]
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        self.shared.value = None
        # Truyền qua lần lượt các khối Transformer từ 0 đến 11
        x = vit.blocks(x)
        x = vit.norm(x)
        return vit.forward_head(x)


def load_ced_tiny(model: ViTSBias, repo: str = CED_REPO, path: str | None = None) -> dict[str, int]:
    """
    Khởi tạo trọng số mô hình từ checkpoint tiền huấn luyện AudioSet chính thức của CED-Tiny.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        local = path or hf_hub_download(repo, "model.safetensors")
        state = load_file(local)
    except Exception as exc:
        print(f"Cảnh báo: Không thể tải trọng số CED-Tiny ({exc}). Sử dụng trọng số khởi tạo ngẫu nhiên.")
        return {"loaded_tensors": 0, "total_tensors": 0, "missing": 0}

    vit = model.vit
    mapped = {k[len(CED_PREFIX) :]: v for k, v in state.items() if k.startswith(CED_PREFIX + "blocks.")}
    mapped.update(
        {
            k[len(CED_PREFIX) :]: v
            for k, v in state.items()
            if k.startswith((CED_PREFIX + "patch_embed.", CED_PREFIX + "norm."))
        }
    )
    own = vit.state_dict()
    mapped = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    result = vit.load_state_dict(mapped, strict=False)

    if (CED_PREFIX + "freq_pos_embed") in state and (CED_PREFIX + "time_pos_embed") in state:
        freq = state[CED_PREFIX + "freq_pos_embed"][0, :, :, 0]
        time = state[CED_PREFIX + "time_pos_embed"][0, :, 0, :]
        if freq.shape[1] != model.freq_patches:
            freq = F.interpolate(freq[None], size=model.freq_patches, mode="linear", align_corners=False)[0]
        time = time[:, : model.time_patches]
        if time.shape[1] >= model.time_patches:
            grid = freq[:, :, None] + time[:, None, :]
            pos = grid.permute(1, 2, 0).reshape(1, model.freq_patches * model.time_patches, -1)
            vit.pos_embed.data[:, 1:] = pos.to(vit.pos_embed.dtype)

    return {"loaded_tensors": len(mapped), "total_tensors": len(own), "missing": len(result.missing_keys)}
