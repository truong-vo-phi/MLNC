"""
Kiểm thử Đơn vị (Unit tests) cho Frontend và 6 Đặc trưng Âm học Vật lý.

Xác minh:
1. Chuẩn hóa năng lượng RMS chính xác về mức target_rms = 0.07925.
2. Kích thước đầu ra STFT: 257 dải tần số và 401 khung thời gian cho 4.0s (64,000 mẫu).
3. 6 đặc trưng âm học (Log-RMS, Crest factor, Kurtosis, Flatness, f5, f6).
4. Kích thước phổ LogMel: [B, 1, 64, 398] kèm chuẩn hóa z-score từng mẫu.
"""

import torch
from src.frontend import (
    TARGET_RMS,
    AcousticFeatureExtractor,
    LogMel,
    SpecAugment,
    rms_normalize,
)


def test_rms_normalization(sample_waveform):
    """Kiểm tra tính bất biến âm lượng: cả sóng âm cực nhỏ và cực lớn đều được chuẩn hóa về 0.07925."""
    wave_quiet = sample_waveform * 0.01
    wave_loud = sample_waveform * 10.0

    norm_quiet = rms_normalize(wave_quiet)
    norm_loud = rms_normalize(wave_loud)

    rms_q = norm_quiet.square().mean(dim=-1).sqrt()
    rms_l = norm_loud.square().mean(dim=-1).sqrt()

    # Cả hai đều phải chuẩn hóa chính xác về mức tham chiếu target_rms = 0.07925
    assert torch.allclose(rms_q, torch.tensor([TARGET_RMS, TARGET_RMS]), atol=1e-4)
    assert torch.allclose(rms_l, torch.tensor([TARGET_RMS, TARGET_RMS]), atol=1e-4)


def test_stft_dimensions_and_frames(sample_waveform):
    """Kiểm tra số dải tần số (257) và số khung thời gian (401) của biến đổi STFT."""
    extractor = AcousticFeatureExtractor(n_fft=512, hop_length=160)
    power = extractor.compute_stft_power(sample_waveform)

    # 512 / 2 + 1 = 257 dải tần số
    # 64000 / 160 + 1 = 401 khung thời gian (kèm center=True)
    assert power.shape == (2, 257, 401)


def test_acoustic_6_features(sample_waveform):
    """Kiểm tra tính đúng đắn về mặt vật lý của 6 đặc trưng âm học."""
    wave_norm = rms_normalize(sample_waveform)
    extractor = AcousticFeatureExtractor(n_fft=512, hop_length=160)
    features = extractor(wave_norm)

    # Kích thước đầu ra phải là [B, 6]
    assert features.shape == (2, 6)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()

    # Kiểm tra tính hợp lệ về mặt vật lý:
    # f1: Log-RMS phải xấp xỉ log(0.07925) ~= -2.535
    f1_log_rms = features[:, 0]
    expected_log_rms = torch.log(torch.tensor(TARGET_RMS))
    assert torch.allclose(f1_log_rms, expected_log_rms.expand(2), atol=1e-2)

    # f2: Hệ số đỉnh Crest factor (peak / rms) phải >= 1.0
    f2_crest = features[:, 1]
    assert (f2_crest >= 1.0).all()

    # f3: Hệ số nhọn Kurtosis của năng lượng phải là số dương
    f3_kurt = features[:, 2]
    assert (f3_kurt > 0.0).all()

    # f4: Độ phẳng phổ (Wiener entropy) phải nằm trong khoảng (0, 1]
    f4_flatness = features[:, 3]
    assert (f4_flatness > 0.0).all()
    assert (f4_flatness <= 1.0 + 1e-4).all()


def test_logmel_and_specaugment(sample_waveform):
    """Kiểm tra tầng LogMel, chuẩn hóa z-score và kỹ thuật SpecAugment."""
    wave_norm = rms_normalize(sample_waveform)
    logmel = LogMel(n_mels=64, num_frames=398)
    mel = logmel(wave_norm)

    # Kích thước phổ đầu ra: [B, 1, 64, 398]
    assert mel.shape == (2, 1, 64, 398)

    # Kiểm tra chuẩn hóa từng mẫu: mean ~= 0, std ~= 1 trên cả thời gian & tần số
    mean = mel.mean(dim=(-2, -1))
    std = mel.std(dim=(-2, -1))
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    # Kiểm tra kỹ thuật SpecAugment
    spec_aug = SpecAugment(freq_masks=2, time_masks=2)
    spec_aug.train()
    aug_mel = spec_aug(mel)
    assert aug_mel.shape == mel.shape
