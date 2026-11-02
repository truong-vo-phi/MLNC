"""
Unit tests for dataset and dataloader modules in Review-3.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import (
    MaskedNoiseCacheDataset,
    SyntheticMaskedNoiseCacheDataset,
    SyntheticMixtureDataset,
    get_dataloader,
    get_dataset,
    rms_normalize,
)


def test_synthetic_mixture_dataset():
    dataset = SyntheticMixtureDataset(num_samples_total=24, num_classes=36)
    assert len(dataset) == 24

    item = dataset[0]
    assert "waveform" in item
    assert "target" in item
    assert "target_idx" in item
    assert "snr_db" in item
    assert "sample_id" in item

    assert item["waveform"].shape == (64000,)
    assert item["target"].shape == (36,)
    assert 0 <= item["target_idx"].item() < 36
    assert item["snr_db"].item() in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def test_rms_normalize():
    # 1D waveform
    wave = torch.randn(64000) * 0.5
    norm_wave = rms_normalize(wave, target_rms=0.07925)
    rms_calc = torch.sqrt(torch.mean(norm_wave ** 2)).item()
    assert abs(rms_calc - 0.07925) < 1e-4

    # 2D batch
    batch_wave = torch.randn(4, 64000) * 0.2
    norm_batch = rms_normalize(batch_wave, target_rms=0.07925)
    batch_rms = torch.sqrt(torch.mean(norm_batch ** 2, dim=-1))
    for r in batch_rms:
        assert abs(r.item() - 0.07925) < 1e-4


def test_masked_noise_cache_dataset_temp_npy():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        npy_file = tmp_path / "validation.npy"

        # Create dummy FRCRN cache of shape (6, 64000)
        rng = np.random.default_rng(42)
        dummy_data = rng.normal(0, 0.1, size=(6, 64000)).astype(np.float32)
        np.save(npy_file, dummy_data)

        # Load with MaskedNoiseCacheDataset
        with MaskedNoiseCacheDataset(
            npy_path=npy_file,
            num_classes=36,
            apply_rms_norm=True,
            target_rms=0.07925,
        ) as ds:
            assert len(ds) == 6

            item = ds[0]
            assert "waveform" in item
            assert "target" in item
            assert "target_idx" in item
            assert "snr_db" in item
            assert "sample_id" in item
            assert item["waveform"].shape == (64000,)
            assert item["target"].shape == (36,)

            # Check RMS normalization was applied
            rms_val = torch.sqrt(torch.mean(item["waveform"] ** 2)).item()
            assert abs(rms_val - 0.07925) < 1e-4


def test_synthetic_masked_noise_cache_dataset():
    ds = SyntheticMaskedNoiseCacheDataset(num_samples_total=18, num_classes=36, apply_rms_norm=True)
    assert len(ds) == 18
    item = ds[0]
    assert item["waveform"].shape == (64000,)
    assert item["target"].shape == (36,)
    rms_val = torch.sqrt(torch.mean(item["waveform"] ** 2)).item()
    assert abs(rms_val - 0.07925) < 1e-4


def test_get_dataset_with_noise_cache():
    # Fallback when path does not exist
    ds = get_dataset(noise_cache_path="non_existent.npy", split="validation")
    assert len(ds) > 0
    item = ds[0]
    assert item["waveform"].shape == (64000,)


def test_dataloader_batching():
    dataset = SyntheticMixtureDataset(num_samples_total=16, num_classes=36)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert batch["waveform"].shape == (4, 64000)
    assert batch["target"].shape == (4, 36)
    assert batch["snr_db"].shape == (4,)
    assert len(batch["sample_id"]) == 4


def test_dataset_factory():
    # Test fallback to synthetic when given non-existent path
    ds = get_dataset(root="non_existent_path_xyz", split="train")
    assert len(ds) > 0

