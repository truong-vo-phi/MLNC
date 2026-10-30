# -*- coding: utf-8 -*-
from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.beat.attention import relative_position_bucket

NUM_BUCKETS = 320     # Số lượng bucket rời rạc cho ma trận khoảng cách tương đối
MAX_DISTANCE = 800    # Khoảng cách thời gian tối đa được mã hóa vị trí
GATE_HIDDEN = 8       # Số chiều ẩn cho cơ chế Gated Relative Position (GREP)
CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


class SharedBias:
    """
    Vùng nhớ đệm chia sẻ bảng Relative Position Bias giữa các khối Transformer.
    Tránh tính toán lại ma trận khoảng cách T x T ở mỗi layer để tiết kiệm bộ nhớ và FLOPs.
    """
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """
    Cơ chế Attention tăng cường với Gated Relative Position Bias (GREP) lấy cảm hứng từ BEATs.
    
    Đặc điểm:
        1. Thay thế Dot-product Attention thuần bằng: (Q @ K^T / sqrt(d)) + Scale * Bias
        2. Bảng embedding vị trí tương đối (relative_attention_bias) được định nghĩa ở Layer 0
           và chia sẻ cho toàn bộ các layer tiếp theo thông qua SharedBias.
        3. Cơ chế cổng GREP (Gated Relative Position): Sử dụng Query để tự động điều tiết mức độ
           ảnh hưởng của vị trí tương đối thông qua cổng phi tuyến tính.
    """
    def __init__(self, base: nn.Module, shared: SharedBias, owns_table: bool, has_cls: bool = True) -> None:
        super().__init__()
        self.has_cls = has_cls
        self.num_heads = base.num_heads
        self.head_dim = base.head_dim
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.shared = shared
        # Chỉ layer đầu tiên (owns_table=True) sở hữu trọng số Embedding bảng vị trí tương đối
        if owns_table:
            self.relative_attention_bias = nn.Embedding(NUM_BUCKETS, self.num_heads)
        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.record = False
        self.cache: dict[str, Tensor] = {}

    def compute_bias(self, tokens: int, device) -> Tensor:
        """
        Tính toán ma trận bias tương đối [1, Num_Heads, Tokens, Tokens].
        Bao gồm cả token CLS (nếu has_cls=True) thông qua padding (1, 0, 1, 0).
        """
        pos = torch.arange(tokens - int(self.has_cls), device=device)
        bucket = relative_position_bucket(pos[None, :] - pos[:, None], NUM_BUCKETS, MAX_DISTANCE)
        bias = self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)
        return F.pad(bias, (1, 0, 1, 0)) if self.has_cls else bias

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        """
        Forward pass của BiasAttention:
            x: Tensor đầu vào [B, T, C] với C = Num_Heads * Head_Dim
        """
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim
        # Chiếu Q, K, V và chuyển về dạng [B, H, T, d]
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        
        # Nếu là layer đầu tiên, tính toán ma trận bias vị trí tương đối và lưu vào shared
        if hasattr(self, "relative_attention_bias"):
            self.shared.value = self.compute_bias(T, x.device)
        position_bias = self.shared.value

        # Cơ chế Cổng GREP: Điều biến biên độ bias theo ngữ cảnh truy vấn Q
        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        
        if self.record:
            self.cache = {"q": q, "k": k, "v": v, "position_bias": position_bias, "scale": scale}
            
        # Áp dụng scale vào ma trận vị trí tương đối
        bias = (scale * position_bias).to(q.dtype)
        
        # Tính attention bằng kernel Flash / Scaled Dot-Product tối ưu phần cứng
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


class AlphaBlock(nn.Module):
    """
    Khối DeepNorm Transformer với hệ số tỷ lệ alpha:
        x = Norm(alpha * x + DropPath(Attention(Norm(x))))
        x = Norm(alpha * x + DropPath(FFN(Norm(x))))
    
    Hệ số alpha = (2 * N)^(1/4) giúp ổn định gradient khi mạng sâu nhiều tầng.
    """
    def __init__(self, block: nn.Module, alpha: float, pre_ln: bool) -> None:
        super().__init__()
        dim = block.norm1.normalized_shape[0]
        self.attn, self.mlp = block.attn, block.mlp
        self.norm1 = block.norm1 if pre_ln else nn.Identity()
        self.norm2 = block.norm2 if pre_ln else nn.Identity()
        self.ls1, self.ls2 = block.ls1, block.ls2
        self.drop_path1, self.drop_path2 = block.drop_path1, block.drop_path2
        self.post_norm1, self.post_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.alpha = alpha

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        x = self.post_norm1(self.alpha * x + self.drop_path1(self.ls1(self.attn(self.norm1(x)))))
        return self.post_norm2(self.alpha * x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))


class ViTSBias(nn.Module):
    """
    Kiến trúc Vision Transformer cho Audio Spectrogram với 2D Relative Bias (CED-Tiny Student).
    
    Cấu trúc mạng:
        - Patch Embedding: Chia phổ Log-Mel (64 bins x 398 frames) thành các patch 16x16.
          Tạo ra 4 patches tần số x 24 patches thời gian = 96 patch tokens.
        - CLS Token: Bổ sung 1 token phân loại toàn cục -> Tổng 97 tokens.
        - 12 Blocks Transformer: Thay thế self-attention bằng BiasAttention.
        - Đầu ra: Classifier Head phân loại 36 lớp âm thanh.
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
        post_ln: bool = False,
        pre_ln: bool = True,
        arch: str = "deit_tiny_patch16_224",
    ) -> None:
        super().__init__()
        # Khởi tạo mô hình DeiT gốc từ thư viện timm
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
        self.freq_patches, self.time_patches = freq_bins // 16, frames // 16
        self.frames = frames
        self.shared = SharedBias()
        alpha = (2 * len(self.vit.blocks)) ** 0.25
        
        # Tiêm BiasAttention vào từng khối Transformer Block
        for index, block in enumerate(self.vit.blocks):
            block.attn = BiasAttention(block.attn, self.shared, owns_table=index == 0)
            if post_ln:
                self.vit.blocks[index] = AlphaBlock(block, alpha, pre_ln)
        self.attentions = [block.attn for block in self.vit.blocks]

    def describe_params(self) -> dict[str, int]:
        """Phân rã chi tiết số lượng tham số theo từng nhóm chức năng"""
        groups = {
            "patch embed (conv 16x16)": 0,
            "CLS + absolute position embedding": 0,
            "attention qkv + proj": 0,
            "bucket + position bias embedding": 0,
            "GRU gate (gated bias)": 0,
            "MLP (fc1 + fc2)": 0,
            "LayerNorm": 0,
            "classifier head": 0,
        }
        for name, param in self.named_parameters():
            n = param.numel()
            if "patch_embed" in name:
                groups["patch embed (conv 16x16)"] += n
            elif name.endswith(("cls_token", "pos_embed")):
                groups["CLS + absolute position embedding"] += n
            elif "relative_attention_bias" in name:
                groups["bucket + position bias embedding"] += n
            elif "grep_" in name:
                groups["GRU gate (gated bias)"] += n
            elif ".attn." in name:
                groups["attention qkv + proj"] += n
            elif ".mlp." in name:
                groups["MLP (fc1 + fc2)"] += n
            elif "head" in name:
                groups["classifier head"] += n
            else:
                groups["LayerNorm"] += n
        return groups

    def set_record(self, enabled: bool) -> None:
        """Kích hoạt ghi lại bộ đệm Q, K, V phục vụ phân tích hoặc chưng cất trung gian"""
        for attention in self.attentions:
            attention.record = enabled
            if not enabled:
                attention.cache = {}

    def init_bias_from_teacher(self, table: Tensor) -> None:
        """Khởi tạo trọng số bảng Relative Bias từ mô hình Teacher BEATs 12L"""
        heads = self.attentions[0].num_heads
        group = table.shape[1] // heads
        grouped = table.reshape(table.shape[0], heads, group).mean(dim=-1)
        self.attentions[0].relative_attention_bias.weight.data.copy_(grouped)

    def forward(self, x: Tensor) -> Tensor:
        """
        Đầu vào: x là phổ Log-Mel chuẩn hóa [B, 1, F, T] (F=64, T=398)
        Đầu ra: Logits phân loại 36 lớp [B, 36]
        """
        vit = self.vit
        # Trích xuất patch tokens [B, N, C]
        x = vit.patch_embed(x[..., : self.frames])
        x = x + vit.pos_embed[:, 1:]
        
        # Chuyển đổi định dạng 2D (tần số x thời gian) để ánh xạ đúng tọa độ
        B, N, C = x.shape
        x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        
        # Thêm token CLS ở đầu chuỗi
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        
        # Reset vùng nhớ chia sẻ bias và duyệt qua 12 khối Transformer
        self.shared.value = None
        x = vit.blocks(x)
        return vit.forward_head(vit.norm(x))


def load_ced_tiny(model: ViTSBias, repo: str = CED_REPO, path: str | None = None) -> dict[str, int]:
    """
    Tải và chuyển đổi trọng số pretrained từ checkpoint CED-Tiny (pretrained trên AudioSet).
    
    Quy trình:
        1. Tải checkpoint safetensors từ HuggingFace Hub hoặc đường dẫn cục bộ.
        2. Tách và ánh xạ các tensor `encoder.blocks.*`, `patch_embed.*`, `norm.*`.
        3. Tái tạo 2D Grid Positional Embedding: Tách riêng `freq_pos_embed` và `time_pos_embed`,
           nội suy kích thước theo số patches thực tế (4 patches tần số x 24 patches thời gian)
           và cộng ma trận ngoài (outer sum) để nạp vào `pos_embed`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    local = path or hf_hub_download(repo, "model.safetensors")
    state = load_file(local)
    vit = model.vit
    norm = state.get(CED_PREFIX + "norm.weight")
    if norm is None:
        raise ValueError(f"{local}: not a CED checkpoint (missing {CED_PREFIX}norm.weight)")
    student_dim = vit.pos_embed.shape[-1]
    if norm.shape[0] != student_dim:
        raise ValueError(
            f"{local}: CED embedding dim {norm.shape[0]} does not match student dim {student_dim}; "
            "use the 192-dim mispeech/ced-tiny checkpoint with deit_tiny_patch16_224"
        )
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

    # Khôi phục và tái tạo Positional Embedding dạng lưới 2D (Tần số + Thời gian)
    freq = state[CED_PREFIX + "freq_pos_embed"][0, :, :, 0]
    time = state[CED_PREFIX + "time_pos_embed"][0, :, 0, :]
    if freq.shape[1] != model.freq_patches:
        freq = F.interpolate(freq[None], size=model.freq_patches, mode="linear", align_corners=False)[0]
    time = time[:, : model.time_patches]
    if time.shape[1] < model.time_patches:
        raise ValueError(f"CED has {time.shape[1]} time positions, student needs {model.time_patches}")
        
    grid = freq[:, :, None] + time[:, None, :]
    pos = grid.permute(1, 2, 0).reshape(1, model.freq_patches * model.time_patches, -1)
    vit.pos_embed.data[:, 1:] = pos.to(vit.pos_embed.dtype)
    return {"loaded_tensors": len(mapped), "total_tensors": len(own), "missing": len(result.missing_keys)}
