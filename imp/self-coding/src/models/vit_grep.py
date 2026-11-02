from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Final

GATE_HIDDEN: Final[int] = 8
CED_REPO: Final[str] = "mispeech/ced-tiny"
CED_PREFIX: Final[str] = "encoder."


class SharedBias:
    """
    Hộp chứa trung gian giúp Block 0 tính ma trận P_rel đúng 1 lần trong mỗi lượt forward,
    và các Block 1..11 chỉ việc đọc lại từ self.shared.value.
    """
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """
    Cơ chế chú ý có điều biến vị trí tương đối 2D (2D Gated Relative Position Bias - GREP).
    - Số head: H = 3
    - Chiều mỗi head: d_k = 64 (Tổng chiều C = 3 * 64 = 192)
    - Chuỗi token: T = 97 (1 CLS + 96 patches từ lưới 4x24)
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
        self.num_heads = base.num_heads  # 3 heads
        self.head_dim = base.head_dim    # 64 chiều
        # Tái sử dụng tầng chiếu QKV và tầng chiếu đầu ra từ timm Attention gốc
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.shared = shared
        self.freq_patches = freq_patches
        self.time_patches = time_patches
        n_patches = freq_patches * time_patches  # 4 * 24 = 96

        # Chỉ Block 0 (owns_table=True) mới khởi tạo bảng tham số vị trí tương đối
        if owns_table:
            num_f = 2 * freq_patches - 1  # 2*4 - 1 = 7 khoảng cách tần số (-3..+3)
            num_t = 2 * time_patches - 1  # 2*24 - 1 = 47 khoảng cách thời gian (-23..+23)
            self.freq_rel_bias = nn.Embedding(num_f, self.num_heads)
            self.time_rel_bias = nn.Embedding(num_t, self.num_heads)
            # Bias riêng cho vị trí của token CLS (hàng 0 và cột 0)
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

        # Cổng GREP riêng cho từng khối: chiếu từ 64 chiều xuống 8 chiều
        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        # Hệ số nhân học được 'a' trong công thức S = g_a * (g_b * a - 1) + 2
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def compute_bias(self, device: torch.device) -> Tensor:
        """
        Tạo ma trận P_rel kích thước [1, 3, 97, 97] từ khoảng cách (delta_f, delta_t) giữa mọi cặp patch.
        """
        f_p, t_p = self.freq_patches, self.time_patches
        # Tọa độ tần số và thời gian của 96 patches
        grid_f = torch.arange(f_p, device=device).unsqueeze(1).expand(f_p, t_p).reshape(-1)
        grid_t = torch.arange(t_p, device=device).unsqueeze(0).expand(f_p, t_p).reshape(-1)

        # Hiệu khoảng cách giữa patch i và patch j, dịch dương để làm chỉ số tra bảng Embedding
        delta_f = (grid_f.unsqueeze(1) - grid_f.unsqueeze(0)) + (f_p - 1)  # [96, 96]
        delta_t = (grid_t.unsqueeze(1) - grid_t.unsqueeze(0)) + (t_p - 1)  # [96, 96]

        # Cộng bias tần số và thời gian: [96, 96, 3] -> [1, 3, 96, 96]
        bias_patch = (self.freq_rel_bias(delta_f) + self.time_rel_bias(delta_t)).permute(2, 0, 1).unsqueeze(0)
        if not self.has_cls:
            return bias_patch

        # Ghép thêm hàng 0 và cột 0 của token CLS -> [1, 3, 97, 97]
        row_cls = torch.cat([self.cls_to_cls, self.cls_to_patches], dim=-1)     # [1, 3, 1, 97]
        rows_patch = torch.cat([self.patches_to_cls, bias_patch], dim=-1)       # [1, 3, 96, 97]
        return torch.cat([row_cls, rows_patch], dim=-2)                         # [1, 3, 97, 97]

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim

        # 1. Tính Q, K, V: mỗi tensor có kích thước [B, 3, 97, 64]
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)

        # 2. Nếu là Block 0, tính P_rel và lưu vào self.shared.value
        if hasattr(self, "freq_rel_bias"):
            self.shared.value = self.compute_bias(x.device)
        position_bias = self.shared.value  # [1, 3, 97, 97]

        # 3. Cổng GREP điều biến biên độ theo ngữ cảnh của Query Q:
        # Linear(64 -> 8) -> gộp nhóm thành 2 cổng -> Sigmoid
        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)  # Mỗi cổng: [B, 3, 97, 1]
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0  # S: [B, 3, 97, 1]

        # 4. Nhân điều biến S với P_rel -> B_attn: [B, 3, 97, 97]
        bias = (scale * position_bias).to(q.dtype) if position_bias is not None else None

        # 5. Scaled Dot-Product Attention: Softmax(QK^T / sqrt(64) + B_attn) * V
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        # Ghép lại 3 heads: [B, 3, 97, 64] -> [B, 97, 192]
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


class ViTSBias(nn.Module):
    """
    Backbone Vision Transformer 12 khối (Pre-LN) tích hợp 2D GREP BiasAttention.
    Mỗi khối có:
      - Nhánh 1: x = x + BiasAttention(LayerNorm1(x))
      - Nhánh 2: x = x + MLP(LayerNorm2(x))  (MLP: 192 -> 768 -> 192 với GELU, 295,872 tham số)
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

        # Thay thế Attention mặc định của cả 12 khối bằng BiasAttention (2D GREP)
        for index, block in enumerate(self.vit.blocks):
            if use_bias:
                block.attn = BiasAttention(
                    block.attn,
                    self.shared,
                    owns_table=(index == 0),  # Chỉ Block 0 giữ bảng
                    freq_patches=self.freq_patches,
                    time_patches=self.time_patches,
                    has_cls=True,
                    bias_init=bias_init,
                )

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        x = vit.patch_embed(x[..., : self.frames])  # [B, 96, 192]
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        if self.time_major:
            x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))  # [B, 97, 192]
        self.shared.value = None
        x = vit.blocks(x)
        x = vit.norm(x)
        return vit.forward_head(x)


def load_ced_tiny(model: ViTSBias, repo: str = CED_REPO, path: str | None = None) -> dict[str, int]:
    """
    Tải trọng số tiền huấn luyện AudioSet của CED-Tiny và nội suy lưới Positional Embedding 2D (4 x 24).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        local = path or hf_hub_download(repo, "model.safetensors")
        state = load_file(local)
    except Exception as exc:
        print(f"Warning: Không tải được CED-Tiny weights ({exc}). Sử dụng trọng số khởi tạo.")
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

    # Nội suy Positional Embedding 2D từ tách biệt tần số (freq_pos_embed) và thời gian (time_pos_embed)
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