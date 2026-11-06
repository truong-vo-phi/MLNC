import torch

from src.frontend import LogMelFrontend


def test_frontend_output_shape_matches_ced_official():
    """4.0 s / 16 kHz must yield exactly 398 frames, like mispeech/ced-tiny."""
    fe = LogMelFrontend()
    out = fe(torch.randn(2, 64000))
    assert out.shape == (2, 1, 64, 398)


def test_frontend_short_crop_is_padded_to_num_frames():
    """Overlap crops (3.2 s) are zero-padded at the end to the model's fixed frame count."""
    fe = LogMelFrontend()
    out = fe(torch.randn(2, 51200))
    assert out.shape == (2, 1, 64, 398)


def test_frontend_accepts_single_waveform():
    fe = LogMelFrontend()
    out = fe(torch.randn(64000))
    assert out.shape == (1, 1, 64, 398)
