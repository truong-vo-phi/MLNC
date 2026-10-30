from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.beat.attention import relative_position_bucket

NUM_BUCKETS = 320
MAX_DISTANCE = 800
GATE_HIDDEN = 8
CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


class SharedBias:
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
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
        bias = self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)
        return F.pad(bias, (1, 0, 1, 0)) if self.has_cls else bias

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
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


class AlphaBlock(nn.Module):
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
        for index, block in enumerate(self.vit.blocks):
            block.attn = BiasAttention(block.attn, self.shared, owns_table=index == 0)
            if post_ln:
                self.vit.blocks[index] = AlphaBlock(block, alpha, pre_ln)
        self.attentions = [block.attn for block in self.vit.blocks]

    def describe_params(self) -> dict[str, int]:
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
        for attention in self.attentions:
            attention.record = enabled
            if not enabled:
                attention.cache = {}

    def init_bias_from_teacher(self, table: Tensor) -> None:
        heads = self.attentions[0].num_heads
        group = table.shape[1] // heads
        grouped = table.reshape(table.shape[0], heads, group).mean(dim=-1)
        self.attentions[0].relative_attention_bias.weight.data.copy_(grouped)

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        x = vit.patch_embed(x[..., : self.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        self.shared.value = None
        x = vit.blocks(x)
        return vit.forward_head(vit.norm(x))


def load_ced_tiny(model: ViTSBias, repo: str = CED_REPO, path: str | None = None) -> dict[str, int]:
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
