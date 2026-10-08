import logging
from types import SimpleNamespace

import pytest
import torch

from src.train import FRCRNTrainConfig, scale_batch_size
from src.utils.batch_size import largest_fitting_batch

CUDA = torch.device("cuda")
TOTAL = 100


@pytest.fixture
def fake_cuda(monkeypatch):
    state = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=TOTAL))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: state.update(peak=0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: state["peak"])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return state


def make_step(state, per_sample, limit=None):
    calls = []

    def step(size):
        calls.append(size)
        if limit is not None and size > limit:
            raise torch.cuda.OutOfMemoryError("out of memory")
        state["peak"] = size * per_sample

    return step, calls


def test_returns_none_off_cuda_or_without_candidates():
    step = lambda size: None  # noqa: E731
    assert largest_fitting_batch(step, [8, 16], torch.device("cpu")) is None
    assert largest_fitting_batch(step, [], CUDA) is None


def test_stops_at_out_of_memory_and_keeps_previous(fake_cuda):
    step, calls = make_step(fake_cuda, per_sample=1, limit=32)
    assert largest_fitting_batch(step, [16, 32, 48, 64], CUDA) == 32
    assert calls == [16, 32, 48]


def test_rejects_sizes_over_vram_fraction(fake_cuda):
    step, calls = make_step(fake_cuda, per_sample=2)
    assert largest_fitting_batch(step, [16, 32, 48, 64], CUDA, max_fraction=0.9) == 32
    assert calls == [16, 32, 48]


def test_none_when_smallest_candidate_fails(fake_cuda):
    step, _ = make_step(fake_cuda, per_sample=1, limit=4)
    assert largest_fitting_batch(step, [8, 16], CUDA) is None


def test_candidates_are_tried_in_ascending_order(fake_cuda):
    step, calls = make_step(fake_cuda, per_sample=1)
    assert largest_fitting_batch(step, [32, 8, 16], CUDA) == 32
    assert calls == [8, 16, 32]


def test_invalid_fraction_raises(fake_cuda):
    with pytest.raises(ValueError):
        largest_fitting_batch(lambda size: None, [8], CUDA, max_fraction=0.0)


def test_scale_batch_size_keeps_config_off_cuda(tmp_path):
    model_yaml = tmp_path / "frcrn.yaml"
    model_yaml.write_text("model:\n  target: speech\n", encoding="utf-8")
    config = FRCRNTrainConfig(batch_size=12, batch_candidates=(16, 32))
    scaled = scale_batch_size(config, model_yaml, torch.device("cpu"), 16000, logging.getLogger("test"))
    assert scaled is config


def test_batch_candidates_load_as_tuple(tmp_path):
    path = tmp_path / "frcrn.yaml"
    path.write_text("train:\n  batch_candidates: [24, 48]\n  vram_fraction: 0.8\n", encoding="utf-8")
    config = FRCRNTrainConfig.from_yaml(path)
    assert config.batch_candidates == (24, 48)
    assert config.vram_fraction == 0.8
