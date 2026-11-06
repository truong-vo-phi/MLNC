"""
Unit tests for dataset and dataloader modules in Dual-CED-Tiny.
"""

import torch

from src.dataset import SyntheticMixtureDataset, get_dataloader, rms_normalize


def test_synthetic_mixture_dataset():
    dataset = SyntheticMixtureDataset(num_samples_total=16, num_classes=36)
    assert len(dataset) == 16

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
    wave = torch.randn(64000) * 0.5
    norm_wave = rms_normalize(wave, target_rms=0.07925)
    rms_calc = torch.sqrt(torch.mean(norm_wave ** 2)).item()
    assert abs(rms_calc - 0.07925) < 1e-4

    batch_wave = torch.randn(4, 64000) * 0.3
    norm_batch = rms_normalize(batch_wave, target_rms=0.07925)
    batch_rms = torch.sqrt(torch.mean(norm_batch ** 2, dim=-1))
    for r in batch_rms:
        assert abs(r.item() - 0.07925) < 1e-4


def test_dataloader_batching():
    dataset = SyntheticMixtureDataset(num_samples_total=16, num_classes=36)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["waveform"].shape == (4, 64000)
    assert batch["target"].shape == (4, 36)
    assert batch["snr_db"].shape == (4,)
