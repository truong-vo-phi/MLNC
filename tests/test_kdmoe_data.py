from types import SimpleNamespace

import numpy as np
import pytest

from scripts.load_dataset import (
    MIXTURE_KEY,
    KDMoEMixDataset,
    PairedClipSampler,
    build_dataloader,
    resolve_dataset_root,
)

GROUP_SIZES = {"E0": 5, "E1": 12, "E2": 9, "E3": 10}


def _dataset_available(key: str = MIXTURE_KEY) -> bool:
    try:
        resolve_dataset_root(key)
    except (KeyError, FileNotFoundError):
        return False
    return True


def _fake_dataset(rows: list[dict[str, str]], labels: list[int]) -> SimpleNamespace:
    return SimpleNamespace(rows=rows, labels=labels)


def test_sampler_units_pair_same_clip_and_cover_all_mixes():
    rows = []
    labels = []
    for clip in range(10):
        for mix in range(3):
            rows.append({"noise_ytid": f"clip{clip}"})
            labels.append(mix % 3)
    dataset = _fake_dataset(rows, labels)
    sampler = PairedClipSampler(dataset, units_per_batch=4, seed=0)
    assert sampler._num_units == 10 * 2
    assert len(sampler) == 20 // 4
    units = sampler._units(np.random.default_rng(0))
    covered = sorted(index for unit in units for index in unit)
    assert covered == list(range(len(rows)))
    for unit in units:
        if len(unit) == 2:
            assert rows[unit[0]]["noise_ytid"] == rows[unit[1]]["noise_ytid"]
        else:
            assert len(unit) == 1


def test_sampler_units_cover_all_mixes():
    rows = [{"noise_ytid": f"clip{clip}"} for clip in range(6) for _ in range(2)]
    labels = [0] * 12
    sampler = PairedClipSampler(_fake_dataset(rows, labels), units_per_batch=3, seed=1)
    seen: list[int] = []
    for _ in range(2):
        for batch in sampler:
            seen.extend(batch)
    assert sorted(seen[:12]) == list(range(12))
    assert sorted(seen[12:]) == list(range(12))


def test_sampler_epoch_changes_with_seed():
    rows = [{"noise_ytid": f"clip{clip}"} for clip in range(8) for _ in range(2)]
    labels = [clip % 4 for clip in range(8) for _ in range(2)]
    sampler = PairedClipSampler(_fake_dataset(rows, labels), units_per_batch=4, seed=0)
    first = [index for batch in sampler for index in batch]
    second = [index for batch in sampler for index in batch]
    assert first != second
    assert sorted(first) == sorted(second) == list(range(16))


@pytest.mark.data
@pytest.mark.skipif(not _dataset_available(), reason="Mixture-36l not available")
def test_kdmoe_dataset_item():
    dataset = KDMoEMixDataset(split="validation")
    item = dataset[0]
    assert item["waveform"].shape == (64000,)
    assert item["noise"].shape == (64000,)
    assert item["target"].shape == (36,)
    assert 0 <= int(item["label"]) < 36
    assert 0 <= int(item["group"]) < 4
    assert isinstance(item["noise_ytid"], str) and item["noise_ytid"]
    assert item["teacher_logits36"].shape == (36,)


@pytest.mark.data
@pytest.mark.skipif(not _dataset_available(), reason="Mixture-36l not available")
def test_paired_sampler_batch_on_real_data():
    dataset = KDMoEMixDataset(split="train")
    sampler = PairedClipSampler(dataset, units_per_batch=32, seed=0)
    loader = build_dataloader(dataset, None, batch_sampler=sampler)
    batch = next(iter(loader))
    ytids = batch["noise_ytid"]
    assert 32 <= len(ytids) <= 64
    assert batch["waveform"].shape[1:] == (64000,)
    assert batch["noise"].shape[1:] == (64000,)
    assert batch["teacher_logits36"].shape[1:] == (36,)
    pairs = sum(ytids[index] == ytids[index + 1] for index in range(len(ytids) - 1))
    assert pairs >= 24
    assert len(set(batch["label"].tolist())) >= 24


@pytest.mark.data
@pytest.mark.skipif(not _dataset_available(), reason="Mixture-36l not available")
def test_label_matches_group_map():
    from scripts.build_group_map import load_group_map

    dataset = KDMoEMixDataset(split="validation")
    group_map = load_group_map()
    mid_to_group = group_map["mid_to_group"]
    checked = 0
    for index in range(0, len(dataset), 500):
        row = dataset.rows[index]
        mid = row["label_mids"].split(",")[0].strip()
        assert dataset.groups[index] == mid_to_group[mid]
        checked += 1
    assert checked > 0


def test_sampler_supports_subset_positions():
    from torch.utils.data import Subset

    rows = [{"noise_ytid": f"clip{clip}"} for clip in range(4) for _ in range(2)]
    labels = [clip % 2 for clip in range(4) for _ in range(2)]
    subset = Subset(_fake_dataset(rows, labels), [0, 1, 4, 5])
    sampler = PairedClipSampler(subset, units_per_batch=2, seed=0)
    units = sampler._units(np.random.default_rng(0))
    covered = sorted(index for unit in units for index in unit)
    assert covered == [0, 1, 2, 3]
    assert len(units) == 2
    for unit in units:
        assert len(unit) == 2
        assert rows[subset.indices[unit[0]]]["noise_ytid"] == rows[subset.indices[unit[1]]]["noise_ytid"]
