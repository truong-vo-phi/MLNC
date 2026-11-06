"""
Unit tests for multi-crop overlap inference.
"""

import torch

from src.inference import overlap_crop_starts, overlap_logits


class _CropMeanModel:
    """Dummy model: returns [mean, -mean] per crop and records the crop lengths it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[float] | None]] = []

    def __call__(self, wave: torch.Tensor, snr_db: torch.Tensor | None = None) -> torch.Tensor:
        self.calls.append((wave.shape[-1], None if snr_db is None else snr_db.tolist()))
        mean = wave.mean(dim=-1)
        return torch.stack([mean, -mean], dim=-1)


def test_overlap_crop_starts_for_4s_clip():
    # 4.0 s clip at 16 kHz, 3 crops of 3.2 s -> starts at 0.0 / 0.4 / 0.8 s.
    assert overlap_crop_starts(64000, num_windows=3, crop_samples=51200) == [0, 6400, 12800]


def test_overlap_crop_starts_includes_both_endpoints():
    starts = overlap_crop_starts(64000, num_windows=5, crop_samples=32000)
    assert starts[0] == 0
    assert starts[-1] == 64000 - 32000
    assert starts == sorted(starts)


def test_overlap_crop_starts_dedupes_when_windows_exceed_positions():
    assert overlap_crop_starts(100, num_windows=20, crop_samples=90) == list(range(0, 11))


def test_overlap_crop_starts_single_window_for_short_clip():
    assert overlap_crop_starts(40000, num_windows=3, crop_samples=51200) == [0]


def test_overlap_logits_averages_three_crops_and_forwards_snr():
    model = _CropMeanModel()
    wave = torch.ones(2, 64000)
    snr = torch.tensor([-5.0, 20.0])

    out = overlap_logits(model, wave, snr_db=snr, num_windows=3, crop_seconds=3.2)

    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([1.0, -1.0]).expand(2, 2))
    assert [length for length, _ in model.calls] == [51200, 51200, 51200]
    assert all(snr_values == [-5.0, 20.0] for _, snr_values in model.calls)


def test_overlap_logits_handles_clip_shorter_than_crop():
    model = _CropMeanModel()
    wave = torch.full((1, 40000), 0.5)

    out = overlap_logits(model, wave, num_windows=3, crop_seconds=3.2)

    assert out.shape == (1, 2)
    assert [length for length, _ in model.calls] == [40000]
    assert torch.allclose(out, torch.tensor([[0.5, -0.5]]))


def test_overlap_logits_rejects_unknown_aggregate():
    model = _CropMeanModel()
    try:
        overlap_logits(model, torch.zeros(1, 64000), aggregate="median")
    except ValueError as exc:
        assert "aggregate" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for unsupported aggregate")
