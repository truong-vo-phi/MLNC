from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from scripts.load_dataset import build_dataloader


def test_cache_loader_covers_every_sample_when_size_is_not_divisible() -> None:
    dataset = TensorDataset(torch.arange(10).float())
    loader = build_dataloader(dataset, 4, False, 0)
    seen = torch.cat([batch[0] for batch in loader]).tolist()
    assert sorted(seen) == list(range(10))
