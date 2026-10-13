import pytest
import torch

from src.model.kdmoe import (
    HybridBlock,
    KDMoEConfig,
    KDMoEModel,
    KDMoEOutput,
    LogMel,
    MaskGate,
)

SIZE_CONFIGS = {
    "S": (
        {
            "d": 128,
            "heads": 4,
            "n_hybrid": 2,
            "n_vit": 2,
            "hybrid_gru_hidden": 64,
            "final_gru_hidden": 64,
            "expert_hidden": 128,
        },
        1245528,
        33024,
    ),
    "M": (
        {
            "d": 256,
            "heads": 8,
            "n_hybrid": 2,
            "n_vit": 4,
            "hybrid_gru_hidden": 128,
            "final_gru_hidden": 128,
            "expert_hidden": 256,
            "drop_path_max": 0.1,
        },
        6380120,
        65792,
    ),
    "L": (
        {
            "d": 384,
            "heads": 6,
            "n_hybrid": 2,
            "n_vit": 4,
            "hybrid_gru_hidden": 192,
            "final_gru_hidden": 192,
            "expert_hidden": 384,
            "drop_path_max": 0.1,
        },
        14214744,
        98560,
    ),
}


def make_model(**overrides) -> KDMoEModel:
    config = KDMoEConfig(d=64, heads=2, n_hybrid=1, n_vit=1, hybrid_gru_hidden=32, final_gru_hidden=16)
    return KDMoEModel(config, **overrides)


@pytest.mark.parametrize("name", ["S", "M", "L"])
def test_param_count_matches_spec(name):
    kwargs, expected, head_expected = SIZE_CONFIGS[name]
    model = KDMoEModel(KDMoEConfig(**kwargs))
    total = sum(parameter.numel() for parameter in model.parameters())
    head = sum(parameter.numel() for parameter in model.noise_head.parameters())
    assert total - head == expected
    assert head == head_expected


def test_forward_shapes():
    model = make_model().eval()
    waveform = torch.randn(2, 64000)
    with torch.no_grad():
        out = model(waveform)
    assert isinstance(out, KDMoEOutput)
    assert out.logits.shape == (2, 36)
    assert out.router_logits.shape == (2, 4)
    assert out.e.shape == (2, 64)
    assert out.gate.shape == (2, 8, 24, 64)
    assert out.noise_mel is None and out.noise_target is None
    assert torch.isfinite(out.logits).all()


def test_forward_with_noise_target_shapes():
    model = make_model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 64000), torch.randn(2, 64000))
    assert out.noise_mel.shape == (2, 1, 128, 384)
    assert out.noise_target.shape == (2, 1, 128, 384)


def test_hybrid_block_alpha_gru_zero_equals_attention_branch():
    torch.manual_seed(0)
    block = HybridBlock(d=32, heads=2, gru_hidden=16, ffn_ratio=2, dropout=0.0, drop_path=0.0).eval()
    with torch.no_grad():
        block.a_gru.zero_()
    x = torch.randn(2, 8, 24, 32)
    with torch.no_grad():
        actual = block(x)
        u = block.ln1(x)
        tokens = u.reshape(2, 8 * 24, 32)
        attended, _ = block.attn(tokens, tokens, tokens, need_weights=False)
        expected = x + block.a_att * attended.reshape(2, 8, 24, 32)
        expected = expected + block.ffn(block.ln2(expected))
    assert torch.allclose(actual, expected, atol=1e-5)


def test_hybrid_block_gru_branch_permutes_with_frequency():
    torch.manual_seed(0)
    block = HybridBlock(d=16, heads=2, gru_hidden=8, ffn_ratio=2, dropout=0.0, drop_path=0.0, branches="gru").eval()
    x = torch.randn(2, 8, 24, 16)
    permutation = torch.randperm(8)
    with torch.no_grad():
        actual = block(x[:, permutation])
        expected = block(x)[:, permutation]
    assert torch.allclose(actual, expected, atol=1e-5)


def test_gate_starts_open_and_never_below_eps():
    eps = 0.05
    torch.manual_seed(0)
    gate = MaskGate(32, eps=eps).eval()
    x = torch.randn(64, 8, 24, 32)
    with torch.no_grad():
        _, g = gate(x)
    assert 0.8 < g.mean().item() < 0.95
    assert (g >= eps).all()


def test_top1_inference_picks_argmax_expert():
    config = KDMoEConfig(
        d=64, heads=2, n_hybrid=1, n_vit=1, hybrid_gru_hidden=32, final_gru_hidden=16, inference_mode="top1"
    )
    model = KDMoEModel(config).eval()
    waveform = torch.randn(2, 64000)
    with torch.no_grad():
        out = model(waveform)
        experts = model.experts(out.e)
        index = out.router_logits.argmax(dim=-1)
        expected = experts.gather(1, index[:, None, None].expand(-1, 1, experts.shape[-1])).squeeze(1)
    assert torch.allclose(out.logits, expected, atol=1e-5)


def test_weighted_training_mixture_experts():
    model = make_model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 64000))
        experts = model.experts(out.e)
        weights = out.router_logits.softmax(dim=-1)
        expected = (weights.unsqueeze(-1) * experts).sum(dim=1)
    assert torch.allclose(out.logits, expected, atol=1e-5)


def test_logmel_crops_and_shares_offset():
    torch.manual_seed(0)
    frontend = LogMel().eval()
    waveform = torch.randn(2, 64000)
    noise = torch.randn(2, 64000)
    with torch.no_grad():
        x, target = frontend(waveform, noise)
        full = frontend.logmel(waveform)
        mean = full.mean(dim=(-2, -1), keepdim=True)
        std = full.std(dim=(-2, -1), keepdim=True)
        normalized = (full - mean) / (std + frontend.norm_eps)
    assert x.shape == (2, 1, 128, 384) and target.shape == (2, 1, 128, 384)
    assert torch.allclose(x.squeeze(1), normalized[:, :, 7 : 7 + 384], atol=1e-5)


def test_config_rejects_unknown_modes():
    with pytest.raises(ValueError):
        KDMoEConfig(hybrid_branches="mlp")
    with pytest.raises(ValueError):
        KDMoEConfig(freq_pool="max")
    with pytest.raises(ValueError):
        KDMoEConfig(inference_mode="top2")
    with pytest.raises(ValueError):
        KDMoEConfig(patch_size=5)


def test_model_from_yaml(tmp_path):
    path = tmp_path / "kdmoe.yaml"
    path.write_text(
        "model:\n  d: 64\n  heads: 2\n  n_hybrid: 1\n  n_vit: 1\n"
        "  hybrid_gru_hidden: 32\n  final_gru_hidden: 16\n  unknown_key: 1\n"
        "frontend:\n  n_mels: 64\n",
        encoding="utf-8",
    )
    model = KDMoEModel.from_yaml(path)
    assert model.config.d == 64
    assert model.frontend.melspec.n_mels == 64
