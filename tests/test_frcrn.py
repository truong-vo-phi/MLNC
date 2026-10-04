import pytest
import torch

from src.loss import FRCRNLoss, si_snr
from src.model.frcrn import FRCRN, FRCRNConfig, FRCRNSeparator
from src.model.frcrn.stft import ConviSTFT, ConvSTFT


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        FRCRNConfig(target="music")


def test_unsupported_fft_len_rejected():
    with pytest.raises(ValueError):
        FRCRN(FRCRNConfig(fft_len=512))


def test_stft_roundtrip():
    config = FRCRNConfig()
    waveform = torch.randn(1, 16000)
    spectrum = ConvSTFT(config)(waveform)
    restored = ConviSTFT(config)(spectrum)
    length = min(restored.shape[-1], waveform.shape[-1]) - config.win_len
    assert torch.allclose(restored[..., config.win_len : length], waveform[..., config.win_len : length], atol=1e-3)


def test_separator_shapes_and_loss():
    separator = FRCRNSeparator(FRCRNConfig())
    mixture = torch.randn(2, 8000) * 0.1
    noise = torch.randn(2, 8000) * 0.05
    assert separator(mixture).shape == mixture.shape
    loss = separator.loss(mixture, noise)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in separator.parameters())


def test_si_snr_is_scale_invariant():
    reference = torch.randn(3, 4000)
    estimate = reference + 0.1 * torch.randn(3, 4000)
    assert torch.allclose(si_snr(reference, estimate), si_snr(reference, 5.0 * estimate), atol=1e-3)


def test_frcrn_loss_zero_mask_error_is_finite():
    config = FRCRNConfig()
    mixture = torch.randn(1, 8000)
    target = mixture * 0.5
    mask = torch.zeros(1, 2 * config.num_bins, (8000 - config.win_len) // config.win_inc + 1)
    assert torch.isfinite(FRCRNLoss(config)(mixture, target, target, mask))
