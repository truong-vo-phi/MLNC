from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

GATE_HIDDEN = 8
CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


class SharedBias:
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
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

        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.record = False
        self.cache: dict[str, Tensor] = {}

    def compute_bias(self, device: torch.device) -> Tensor:
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
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        if hasattr(self, "freq_rel_bias"):
            self.shared.value = self.compute_bias(x.device)
        position_bias = self.shared.value
        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        if self.record:
            self.cache = {"q": q, "k": k, "v": v, "position_bias": position_bias, "scale": scale}
        bias = (scale * position_bias).to(q.dtype) if position_bias is not None else None
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
        bias_init: str = "zero",
        use_bias: bool = True,
        time_major: bool = False,
        hybrid_pool: bool = False,
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.time_major = time_major
        self.hybrid_pool = hybrid_pool
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
        self.freq_patches = freq_bins // 16
        self.time_patches = frames // 16
        self.frames = frames
        self.shared = SharedBias()
        alpha = (2 * len(self.vit.blocks)) ** 0.25
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
            if post_ln:
                self.vit.blocks[index] = AlphaBlock(block, alpha, pre_ln)
        self.attentions = [block.attn for block in self.vit.blocks if isinstance(block.attn, BiasAttention)]

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
            elif any(k in name for k in ("rel_bias", "relative_attention_bias", "cls_to_", "patches_to_")):
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

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        x = vit.patch_embed(x[..., : self.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        if self.time_major:
            x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        self.shared.value = None
        x = vit.blocks(x)
        x = vit.norm(x)
        if self.hybrid_pool:
            pooled = 0.5 * (x[:, 0] + x[:, 1:].mean(dim=1))
            return vit.head(pooled)
        return vit.forward_head(x)


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
