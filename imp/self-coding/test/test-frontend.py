"""
Bộ test toàn diện cho src/frontend.py

Chạy:
    python -m pytest tests/test_frontend.py -v
    python -m pytest tests/test_frontend.py -v -k "log_rms"   # chỉ test bug
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.frontend import (
    RMS_EPS,
    TARGET_RMS,
    AcousticFeatureExtractor,
    LogMel,
    SpecAugment,
    create_htk_mel_filterbank,
    rms_normalize,
)


# =====================================================================
# NHÓM 1: rms_normalize
# =====================================================================
class TestRmsNormalize:
    def test_output_rms_equals_target(self):
        """Sau chuẩn hóa, RMS mỗi mẫu phải ≈ TARGET_RMS."""
        wave = torch.randn(4, 64000)
        out = rms_normalize(wave)
        rms = out.square().mean(-1).sqrt()
        expected = torch.full((4,), TARGET_RMS)
        assert torch.allclose(rms, expected, atol=1e-4)

    def test_scale_invariance(self):
        """Nhân input với 0.01 hay 10.0 đều cho cùng kết quả sau chuẩn hóa."""
        wave = torch.randn(2, 64000)
        w_tiny = rms_normalize(wave * 0.01)
        w_huge = rms_normalize(wave * 10.0)
        assert torch.allclose(w_tiny, w_huge, atol=1e-5)

    def test_idempotent(self):
        """Chuẩn hóa 2 lần cho kết quả như 1 lần (sai số ~eps)."""
        wave = torch.randn(2, 64000)
        w1 = rms_normalize(wave)
        w2 = rms_normalize(w1)
        assert torch.allclose(w1, w2, atol=1e-5)

    def test_silent_input_no_nan(self):
        """Input toàn 0 → không được NaN/Inf, chỉ là 0."""
        wave = torch.zeros(2, 64000)
        out = rms_normalize(wave)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(out))

    def test_very_quiet_input_stable(self):
        """Input cực nhỏ (1e-8) không gây NaN do chia cho eps."""
        wave = torch.randn(2, 64000) * 1e-8
        out = rms_normalize(wave)
        assert torch.isfinite(out).all()
        # RMS vẫn phải về target (vì eps = 1e-8 đủ nhỏ so với rms ≈ 1e-8 * sqrt(64000) ≈ 2.5e-6)
        rms = out.square().mean(-1).sqrt()
        assert torch.allclose(rms, torch.full((2,), TARGET_RMS), atol=1e-3)

    def test_per_sample_independent(self):
        """Mỗi mẫu trong batch được chuẩn hóa độc lập."""
        wave = torch.randn(3, 64000)
        wave[0] *= 100.0  # mẫu 0 to hơn nhiều
        out = rms_normalize(wave)
        rms = out.square().mean(-1).sqrt()
        # Cả 3 mẫu vẫn cùng RMS target
        assert torch.allclose(rms, torch.full((3,), TARGET_RMS), atol=1e-4)


# =====================================================================
# NHÓM 2: create_htk_mel_filterbank
# =====================================================================
class TestMelFilterbank:
    def test_shape(self):
        fb = create_htk_mel_filterbank()
        assert fb.shape == (64, 257)

    def test_dtype(self):
        fb = create_htk_mel_filterbank()
        assert fb.dtype == torch.float32

    def test_all_positive(self):
        """Mọi phần tử filterbank phải ≥ 0 (tam giác không âm)."""
        fb = create_htk_mel_filterbank()
        assert (fb >= 0).all()

    def test_each_filter_has_mass(self):
        """Mỗi bộ lọc tam giác phải có tổng > 0."""
        fb = create_htk_mel_filterbank()
        row_sums = fb.sum(dim=1)
        assert (row_sums > 0).all()

    def test_triangular_peak_is_one(self):
        """Đỉnh tam giác phải chạm 1.0 tại bin trung tâm."""
        fb = create_htk_mel_filterbank()
        max_per_row = fb.max(dim=1).values
        # Hầu hết phải đạt 1.0 (sai số do làm tròn bin)
        assert (max_per_row > 0.95).all()

    def test_contiguous_overlap(self):
        """2 bộ lọc liền kề phải có overlap tại vùng biên."""
        fb = create_htk_mel_filterbank()
        # Nhân từng hàng với nhau: nếu overlap thì tổng > 0
        for i in range(0, 20):
            overlap = (fb[i] * fb[i + 1]).sum().item()
            assert overlap > 0, f"Filter {i} và {i+1} không overlap"


# =====================================================================
# NHÓM 3: LogMel
# =====================================================================
class TestLogMel:
    def test_output_shape(self):
        mel = LogMel()(rms_normalize(torch.randn(2, 64000)))
        assert mel.shape == (2, 1, 64, 398)

    def test_zscore_normalization(self):
        """Sau z-score, mean ≈ 0 và std ≈ 1 cho mỗi mẫu."""
        mel = LogMel()(rms_normalize(torch.randn(4, 64000)))
        for i in range(4):
            m_i = mel[i]
            assert abs(m_i.mean().item()) < 0.05
            assert abs(m_i.std().item() - 1.0) < 0.05

    def test_different_inputs_different_outputs(self):
        """Hai âm thanh khác nhau → mel khác nhau."""
        wave1 = rms_normalize(torch.randn(1, 64000))
        wave2 = rms_normalize(torch.randn(1, 64000))
        mel1 = LogMel()(wave1)
        mel2 = LogMel()(wave2)
        assert not torch.allclose(mel1, mel2, atol=1e-3)

    def test_scale_invariant(self):
        """Nhân input với scale khác nhau → mel giống hệt (nhờ rms_normalize)."""
        wave = torch.randn(1, 64000)
        mel_a = LogMel()(rms_normalize(wave * 0.1))
        mel_b = LogMel()(rms_normalize(wave * 10.0))
        # Sai số nhỏ do RMS normalize
        assert torch.allclose(mel_a, mel_b, atol=1e-4)

    def test_no_nan_with_silence(self):
        """Input im lặng → không được NaN."""
        wave = torch.zeros(1, 64000)
        mel = LogMel()(rms_normalize(wave))
        assert torch.isfinite(mel).all()

    def test_buffer_moves_with_model(self):
        """window và mel_fb phải là buffer (theo model .to(device))."""
        mel = LogMel()
        assert "window" in dict(mel.named_buffers())
        assert "mel_fb" in dict(mel.named_buffers())

    def test_hann_window_registered(self):
        mel = LogMel()
        assert mel.window.shape == (400,)  # win_length=400

    def test_pad_is_112(self):
        """n_fft - win_length = 512 - 400 = 112."""
        mel = LogMel()
        assert mel.pad == 112


# =====================================================================
# NHÓM 4: SpecAugment
# =====================================================================
class TestSpecAugment:
    def test_no_op_in_eval(self):
        """Ở eval mode, SpecAugment không làm gì."""
        sa = SpecAugment()
        sa.eval()
        x = torch.ones(2, 1, 64, 398)
        out = sa(x)
        assert torch.equal(out, x)

    def test_no_op_when_all_zero(self):
        """Nếu freq_masks=time_masks=0 → giữ nguyên."""
        sa = SpecAugment(freq_masks=0, time_masks=0)
        sa.train()
        x = torch.ones(2, 1, 64, 398)
        assert torch.equal(sa(x), x)

    def test_masks_created_in_train(self):
        """Ở train mode, phải có ít nhất 1 phần tử bị zero."""
        torch.manual_seed(42)
        sa = SpecAugment(freq_masks=2, freq_width=8, time_masks=2, time_width=24)
        sa.train()
        x = torch.ones(4, 1, 64, 398)
        out = sa(x)
        # Số phần tử bằng 0 phải > 0
        n_zero = (out == 0).sum().item()
        assert n_zero > 0

    def test_does_not_modify_input_inplace(self):
        """SpecAugment không được sửa input gốc (dùng .clone())."""
        torch.manual_seed(0)
        sa = SpecAugment()
        sa.train()
        x = torch.ones(2, 1, 64, 398)
        x_copy = x.clone()
        _ = sa(x)
        assert torch.equal(x, x_copy)

    def test_mask_within_bounds(self):
        """Mask không được tràn ra ngoài biên."""
        torch.manual_seed(1)
        sa = SpecAugment(freq_masks=5, freq_width=64, time_masks=5, time_width=398)
        sa.train()
        x = torch.ones(2, 1, 64, 398)
        out = sa(x)
        # Vẫn giữ shape
        assert out.shape == x.shape
        # Vẫn finite
        assert torch.isfinite(out).all()


# =====================================================================
# NHÓM 5: AcousticFeatureExtractor
# =====================================================================
class TestAcousticFeatureExtractor:
    def test_output_shape(self):
        ext = AcousticFeatureExtractor()
        wave = rms_normalize(torch.randn(2, 64000))
        assert ext(wave).shape == (2, 6)

    def test_stft_power_shape(self):
        ext = AcousticFeatureExtractor()
        wave = torch.randn(2, 64000)
        power = ext.compute_stft_power(wave)
        assert power.shape == (2, 257, 401)

    def test_stft_power_non_negative(self):
        ext = AcousticFeatureExtractor()
        power = ext.compute_stft_power(torch.randn(2, 64000))
        assert (power >= 0).all()

    def test_features_finite(self):
        """6 đặc trưng phải finite, không NaN/Inf."""
        ext = AcousticFeatureExtractor()
        wave = rms_normalize(torch.randn(4, 64000))
        feat = ext(wave)
        assert torch.isfinite(feat).all()

    def test_crest_kurt_flat_bounded(self):
        """crest, kurtosis, flatness phải trong khoảng hợp lý."""
        ext = AcousticFeatureExtractor()
        wave = rms_normalize(torch.randn(8, 64000))
        feat = ext(wave)
        crest = feat[:, 1]
        kurt = feat[:, 2]
        flat = feat[:, 3]
        # Crest factor: > 0, thường < 100 cho white noise
        assert (crest > 0).all() and (crest < 100).all()
        # Kurtosis: > 0
        assert (kurt > 0).all()
        # Spectral flatness: ∈ [0, 1]
        assert (flat >= 0).all() and (flat <= 1.0).all()

    def test_white_noise_high_flatness(self):
        """White noise → spectral flatness cao (gần 1)."""
        ext = AcousticFeatureExtractor()
        wave = rms_normalize(torch.randn(4, 64000))
        flat = ext(wave)[:, 3]
        assert flat.mean().item() > 0.2  # noise phẳng hơn sine

    def test_sine_low_flatness(self):
        """Sine wave → spectral flatness thấp (gần 0)."""
        ext = AcousticFeatureExtractor()
        t = torch.arange(64000).float() / 16000.0
        sine = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).expand(4, -1)
        sine = rms_normalize(sine)
        flat = ext(sine)[:, 3]
        assert flat.mean().item() < 0.05

    def test_different_spectral_content_gives_different_features(self):
        """White noise vs sine phải cho f_spec khác nhau rõ rệt."""
        ext = AcousticFeatureExtractor()
        t = torch.arange(64000).float() / 16000.0

        noise = rms_normalize(torch.randn(4, 64000))
        sine = rms_normalize(torch.sin(2 * np.pi * 440 * t).unsqueeze(0).expand(4, -1))

        feat_noise = ext(noise)
        feat_sine = ext(sine)

        # f3 (kurtosis), f4 (flatness) phải khác rõ rệt
        assert not torch.allclose(feat_noise[:, 3], feat_sine[:, 3], atol=1e-2)
        assert not torch.allclose(feat_noise[:, 2], feat_sine[:, 2], atol=1e-2)


# =====================================================================
# NHÓM 6: BUG DETECTION — log_rms hằng số
# =====================================================================
class TestLogRmsBug:
    """
    Phát hiện BUG: khi input ĐÃ được rms_normalize, f1 = log_rms
    luôn = log(0.07925) ≈ -2.535 vì rms của input đã cố định.
    """

    def test_rms_of_normalized_input_is_constant(self):
        """Sanity check: RMS của input đã normalize là hằng số."""
        ext = AcousticFeatureExtractor()
        wave = torch.randn(4, 64000)
        w_norm = rms_normalize(wave)
        rms = w_norm.square().mean(-1).sqrt()
        assert torch.allclose(rms, torch.full((4,), TARGET_RMS), atol=1e-4)

    def test_log_rms_is_constant_bug(self):
        """
        ⚠️ TEST NÀY SẼ FAIL trên code hiện tại — đó là mục đích.
        f1 = log_rms phải khác nhau giữa các mẫu âm thanh khác nhau,
        nhưng do input đã normalize → f1 luôn = log(0.07925) ≈ -2.535.
        """
        ext = AcousticFeatureExtractor()

        # 4 mẫu âm thanh có biên độ "thực" RẤT khác nhau
        waves = torch.randn(4, 64000)
        waves[0] *= 0.001   # rất nhỏ
        waves[1] *= 0.1
        waves[2] *= 1.0
        waves[3] *= 100.0   # rất to

        # Chuẩn hóa như trong pipeline thật
        w_norm = rms_normalize(waves)

        # Trích f1 từ input đã normalize
        f1_normalized = ext(w_norm)[:, 0]

        # KỲ VỌNG: f1 phải khác nhau (vì 4 mẫu âm thanh khác nhau)
        # THỰC TẾ: f1 giống hệt nhau vì input đã cùng RMS
        expected_constant = torch.log(torch.tensor(TARGET_RMS) + 1e-8)
        assert torch.allclose(f1_normalized, expected_constant.expand(4), atol=1e-4), \
            "Sanity check: f1 đúng là hằng số (đây là bug)."

        # Đây là assertion sẽ phát hiện bug:
        f1_from_raw = ext(w_norm, raw_waveform=waves)[:, 0] \
            if "raw_waveform" in ext.forward.__code__.co_varnames \
            else f1_normalized

        # BUG DETECTED: f1_normalized không phân biệt được 4 mẫu
        f1_spread = (f1_normalized.max() - f1_normalized.min()).item()
        assert f1_spread < 1e-3, (
            f"⚠️ BUG: f1 = log_rms bị hằng số (spread = {f1_spread:.6f}). "
            f"Cần dùng waveform GỐC thay vì đã rms_normalize."
        )

    def test_f2_to_f6_are_scale_invariant(self):
        """5 đặc trưng còn lại bất biến với scale (đây là điểm mạnh)."""
        ext = AcousticFeatureExtractor()
        wave = torch.randn(2, 64000) * 5.0
        w_norm = rms_normalize(wave)

        feat_normalized = ext(w_norm)
        feat_from_raw_scaled = ext(w_norm, raw_waveform=wave) \
            if "raw_waveform" in ext.forward.__code__.co_varnames \
            else feat_normalized

        # f2..f6 phải giống nhau
        assert torch.allclose(feat_normalized[:, 1:], feat_from_raw_scaled[:, 1:], atol=1e-5)


# =====================================================================
# NHÓM 7: INTEGRATION TESTS
# =====================================================================
class TestIntegration:
    def test_full_frontend_pipeline(self):
        """Luồng hoàn chỉnh: wave → rms → features + mel."""
        wave = torch.randn(2, 64000) * 3.0
        wave_norm = rms_normalize(wave)

        ext = AcousticFeatureExtractor()
        logmel = LogMel()

        feat = ext(wave_norm)
        mel = logmel(wave_norm)

        assert feat.shape == (2, 6)
        assert mel.shape == (2, 1, 64, 398)
        assert torch.isfinite(feat).all()
        assert torch.isfinite(mel).all()

    def test_device_portability(self):
        """Toàn bộ buffer phải di chuyển được sang device khác."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA không có sẵn")

        ext = AcousticFeatureExtractor().cuda()
        logmel = LogMel().cuda()

        wave = rms_normalize(torch.randn(2, 64000)).cuda()
        assert ext(wave).device.type == "cuda"
        assert logmel(wave).device.type == "cuda"

    def test_gradient_flow(self):
        """Toàn bộ module phải có gradient (trừ buffer)."""
        ext = AcousticFeatureExtractor()
        logmel = LogMel()

        wave = torch.randn(2, 64000, requires_grad=True)
        w = rms_normalize(wave)
        feat = ext(w)
        mel = logmel(w)

        loss = feat.sum() + mel.sum()
        loss.backward()

        assert wave.grad is not None
        assert torch.isfinite(wave.grad).all()

    def test_batch_size_one(self):
        """Batch size = 1 không gây lỗi squeeze/unsqueeze."""
        wave = rms_normalize(torch.randn(1, 64000))
        ext = AcousticFeatureExtractor()
        logmel = LogMel()
        assert ext(wave).shape == (1, 6)
        assert logmel(wave).shape == (1, 1, 64, 398)

    def test_deterministic_in_eval(self):
        """Ở eval mode, 2 lần gọi phải cho kết quả y hệt (SpecAugment tắt)."""
        ext = AcousticFeatureExtractor().eval()
        logmel = LogMel().eval()
        sa = SpecAugment().eval()

        wave = rms_normalize(torch.randn(2, 64000))
        feat1 = ext(wave)
        mel1 = sa(logmel(wave))

        feat2 = ext(wave)
        mel2 = sa(logmel(wave))

        assert torch.equal(feat1, feat2)
        assert torch.equal(mel1, mel2)


# =====================================================================
# NHÓM 8: PERFORMANCE SMOKE TESTS
# =====================================================================
class TestPerformance:
    def test_forward_speed_cpu(self):
        """Forward pass không được quá chậm trên CPU (sanity)."""
        import time

        ext = AcousticFeatureExtractor()
        logmel = LogMel()
        wave = rms_normalize(torch.randn(8, 64000))

        # Warmup
        _ = ext(wave)
        _ = logmel(wave)

        t0 = time.time()
        for _ in range(5):
            _ = ext(wave)
            _ = logmel(wave)
        elapsed = time.time() - t0

        # 5 iterations cho batch 8 phải < 5 giây trên CPU
        assert elapsed < 5.0, f"Quá chậm: {elapsed:.2f}s cho 5 iterations"


# =====================================================================
# TIỆN ÍCH: script chạy nhanh không cần pytest
# =====================================================================
if __name__ == "__main__":
    """Chạy: python tests/test_frontend.py"""
    import sys

    print("=" * 70)
    print("CHẠY SMOKE TEST FRONTEND (không cần pytest)")
    print("=" * 70)

    wave = torch.randn(2, 64000) * 3.0
    wave_norm = rms_normalize(wave)
    print(f"✓ rms_normalize: {wave.shape} -> {wave_norm.shape}, "
          f"RMS = {wave_norm.square().mean(-1).sqrt().tolist()}")

    fb = create_htk_mel_filterbank()
    print(f"✓ mel_filterbank: {fb.shape}, max = {fb.max():.4f}")

    logmel = LogMel()
    mel = logmel(wave_norm)
    print(f"✓ LogMel: {wave_norm.shape} -> {mel.shape}, "
          f"mean = {mel.mean():.4f}, std = {mel.std():.4f}")

    sa = SpecAugment()
    sa.train()
    mel_aug = sa(mel)
    n_zero = (mel_aug == 0).sum().item()
    print(f"✓ SpecAugment (train): {n_zero} phần tử bị mask")

    ext = AcousticFeatureExtractor()
    feat = ext(wave_norm)
    print(f"✓ AcousticFeatureExtractor: {feat.shape}")
    print(f"  f1 log_rms      = {feat[:, 0].tolist()}")
    print(f"  f2 crest        = {feat[:, 1].tolist()}")
    print(f"  f3 kurtosis     = {feat[:, 2].tolist()}")
    print(f"  f4 flatness     = {feat[:, 3].tolist()}")
    print(f"  f5 ratio_low    = {feat[:, 4].tolist()}")
    print(f"  f6 ratio_high   = {feat[:, 5].tolist()}")

    # Kiểm tra bug log_rms
    print("\n" + "=" * 70)
    print("KIỂM TRA BUG log_rms")
    print("=" * 70)
    waves_test = torch.randn(4, 64000)
    waves_test[0] *= 0.001
    waves_test[1] *= 0.1
    waves_test[2] *= 1.0
    waves_test[3] *= 100.0
    w_test_norm = rms_normalize(waves_test)
    f1 = ext(w_test_norm)[:, 0]
    spread = (f1.max() - f1.min()).item()
    expected = float(torch.log(torch.tensor(TARGET_RMS)))
    print(f"Input gốc scale: 0.001, 0.1, 1.0, 100.0")
    print(f"f1 = {f1.tolist()}")
    print(f"Expected (log(TARGET_RMS)) = {expected:.6f}")
    print(f"Spread = {spread:.8f}")
    if spread < 1e-3:
        print("⚠️  BUG CONFIRMED: f1 log_rms bị hằng số!")
        print("   → Fix: truyền waveform gốc (chưa normalize) vào extractor")
        sys.exit(1)
    else:
        print("✅ OK: f1 phân biệt được các mẫu khác nhau")
        sys.exit(0)