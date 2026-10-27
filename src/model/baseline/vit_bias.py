from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.beat.attention import relative_position_bucket

NUM_BUCKETS = 320
MAX_DISTANCE = 800
GATE_HIDDEN = 8


class SharedBias:
    """Bias vị trí do khối 1 tạo ra, các khối sau dùng lại (như BEATs)."""

    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """Attention của timm (giữ `qkv`, `proj` gốc) cộng bias vị trí tương đối có cổng kiểu BEATs.

    Token 0 là CLS, các token còn lại xếp theo (thời gian, tần số) như BEATs; CLS có bias bằng 0.
    """

    def __init__(self, base: nn.Module, shared: SharedBias, owns_table: bool, has_cls: bool = True) -> None:
        super().__init__()
        self.has_cls = has_cls
        self.num_heads = base.num_heads
        self.head_dim = base.head_dim
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.shared = shared
        if owns_table:
            self.relative_attention_bias = nn.Embedding(NUM_BUCKETS, self.num_heads)
        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.record = False
        self.cache: dict[str, Tensor] = {}

    def compute_bias(self, tokens: int, device) -> Tensor:
        pos = torch.arange(tokens - int(self.has_cls), device=device)
        bucket = relative_position_bucket(pos[None, :] - pos[:, None], NUM_BUCKETS, MAX_DISTANCE)
        bias = self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        return F.pad(bias, (1, 0, 1, 0)) if self.has_cls else bias  # CLS: bias 0 với mọi token

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        if hasattr(self, "relative_attention_bias"):
            self.shared.value = self.compute_bias(T, x.device)
        position_bias = self.shared.value
        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        if self.record:
            self.cache = {"q": q, "k": k, "v": v, "position_bias": position_bias, "scale": scale}
        bias = (scale * position_bias).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


class AlphaBlock(nn.Module):
    """Khối kiểu BEATs trên ViT-S: LN(alpha * x + sublayer), tuỳ chọn giữ LayerNorm 1 và 2 (pre-LN) trước sublayer."""

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
    """ViT-S/16 (timm) với attention có bias vị trí tương đối có cổng, đầu vào log-mel (B, 1, 128, >=384)."""

    def __init__(self, num_classes: int = 36, freq_bins: int = 128, frames: int = 384, pretrained: bool = True,
                 dropout: float = 0.1, attn_dropout: float = 0.1, drop_path: float = 0.1,
                 post_ln: bool = False, pre_ln: bool = True, arch: str = "vit_small_patch16_224") -> None:
        super().__init__()
        self.vit = timm.create_model(
            arch, pretrained=pretrained, in_chans=1, num_classes=num_classes,
            img_size=(freq_bins, frames), drop_rate=dropout, proj_drop_rate=dropout, attn_drop_rate=attn_dropout,
            drop_path_rate=drop_path,
        )
        self.freq_patches, self.time_patches = freq_bins // 16, frames // 16
        self.frames = frames
        self.shared = SharedBias()
        alpha = (2 * len(self.vit.blocks)) ** 0.25  # DeepNorm như BEATs: (2N)^0,25
        for index, block in enumerate(self.vit.blocks):
            block.attn = BiasAttention(block.attn, self.shared, owns_table=index == 0)
            if post_ln:
                self.vit.blocks[index] = AlphaBlock(block, alpha, pre_ln)
        self.attentions = [block.attn for block in self.vit.blocks]

    def describe_params(self) -> dict[str, int]:
        """Số tham số theo nhóm (đã gộp 12 khối)."""
        groups = {"patch embed (conv 16x16)": 0, "CLS + embedding vị trí tuyệt đối": 0, "attention qkv + proj": 0,
                  "bucket + embedding (bias vị trí)": 0, "cổng GRU (bias có cổng)": 0, "MLP (fc1 + fc2)": 0,
                  "LayerNorm": 0, "đầu phân loại": 0}
        for name, param in self.named_parameters():
            n = param.numel()
            if "patch_embed" in name:
                groups["patch embed (conv 16x16)"] += n
            elif name.endswith(("cls_token", "pos_embed")):
                groups["CLS + embedding vị trí tuyệt đối"] += n
            elif "relative_attention_bias" in name:
                groups["bucket + embedding (bias vị trí)"] += n
            elif "grep_" in name:
                groups["cổng GRU (bias có cổng)"] += n
            elif ".attn." in name:
                groups["attention qkv + proj"] += n
            elif ".mlp." in name:
                groups["MLP (fc1 + fc2)"] += n
            elif "head" in name:
                groups["đầu phân loại"] += n
            else:
                groups["LayerNorm"] += n
        return groups

    def set_record(self, enabled: bool) -> None:
        for attention in self.attentions:
            attention.record = enabled
            if not enabled:
                attention.cache = {}

    def init_bias_from_teacher(self, table: Tensor) -> None:
        """table (320, H_teacher): nhóm trung bình các head liền kề của teacher thành head của student."""
        heads = self.attentions[0].num_heads
        group = table.shape[1] // heads
        grouped = table.reshape(table.shape[0], heads, group).mean(dim=-1)
        self.attentions[0].relative_attention_bias.weight.data.copy_(grouped)

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        x = vit.patch_embed(x[..., : self.frames])  # (B, F*T, C), thứ tự (tần số, thời gian)
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)  # (thời gian, tần số)
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        self.shared.value = None
        x = vit.blocks(x)
        return vit.forward_head(vit.norm(x))


CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


def load_ced_tiny(model: ViTSBias, repo: str = CED_REPO) -> dict[str, int]:
    """Nạp trọng số CED-Tiny (AudioSet) vào student DeiT-Ti: khối, patch embed, LayerNorm cuối và embedding vị trí.

    CED dùng embedding vị trí tách theo tần số (1, C, 4, 1) và thời gian (1, C, 1, 63), không có CLS và đầu 527 lớp.
    Ở đây embedding vị trí của từng patch = tần số + thời gian (tần số nội suy nếu số dải khác 4, thời gian lấy phần đầu);
    CLS và đầu phân loại giữ khởi tạo mới. CED huấn luyện trên 64 mel nên student nên dùng n_mels = 64.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    state = load_file(hf_hub_download(repo, "model.safetensors"))
    vit = model.vit
    mapped = {k[len(CED_PREFIX) :]: v for k, v in state.items() if k.startswith(CED_PREFIX + "blocks.")}
    mapped.update({k[len(CED_PREFIX) :]: v for k, v in state.items() if k.startswith((CED_PREFIX + "patch_embed.", CED_PREFIX + "norm."))})
    own = vit.state_dict()
    mapped = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    result = vit.load_state_dict(mapped, strict=False)

    freq = state[CED_PREFIX + "freq_pos_embed"][0, :, :, 0]  # (C, F_ced)
    time = state[CED_PREFIX + "time_pos_embed"][0, :, 0, :]  # (C, T_ced)
    if freq.shape[1] != model.freq_patches:
        freq = F.interpolate(freq[None], size=model.freq_patches, mode="linear", align_corners=False)[0]
    time = time[:, : model.time_patches]
    if time.shape[1] < model.time_patches:
        raise ValueError(f"CED has {time.shape[1]} time positions, student needs {model.time_patches}")
    grid = freq[:, :, None] + time[:, None, :]  # (C, F, T)
    pos = grid.permute(1, 2, 0).reshape(1, model.freq_patches * model.time_patches, -1)  # thứ tự (tần số, thời gian)
    vit.pos_embed.data[:, 1:] = pos.to(vit.pos_embed.dtype)
    return {"loaded_tensors": len(mapped), "total_tensors": len(own), "missing": len(result.missing_keys)}
