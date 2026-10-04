import pytest
import torch

from src.model.beat import BEATClassifier, BEATConfig


@pytest.fixture(scope="module")
def tiny_model():
    config = BEATConfig(
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=128,
        conv_pos=16,
        conv_pos_groups=4,
        num_classes=5,
    )
    return BEATClassifier(config).eval()


def test_forward_shapes(tiny_model):
    fbank = torch.randn(2, 64, 128)
    with torch.no_grad():
        out = tiny_model(fbank, is_waveform=False)
    assert out.logits.shape == (2, 5)
    assert out.features.shape == (2, (64 // 16) * 8, 64)
    assert torch.isfinite(out.logits).all()


def test_config_from_yaml(tmp_path):
    path = tmp_path / "beat.yaml"
    path.write_text("model:\n  encoder_layers: 3\n  num_classes: 7\n  unknown_key: 1\n", encoding="utf-8")
    config = BEATConfig.from_yaml(path)
    assert config.encoder_layers == 3
    assert config.num_classes == 7


def test_default_param_count_matches_beats_base():
    model = BEATClassifier(BEATConfig())
    millions = sum(p.numel() for p in model.parameters()) / 1e6
    assert 89 < millions < 92
