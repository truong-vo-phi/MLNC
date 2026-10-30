from __future__ import annotations

import numpy as np
import torch
from ban_beat.runtime import rms_normalize
from ban_beat.teacher import load_teacher_cache, save_teacher_cache, teacher_fingerprint


def test_teacher_cache_roundtrip(tmp_path) -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]], dtype=np.float32)
    path = save_teacher_cache(tmp_path / "teacher_train.npz", ["a", "b"], logits, "fp")
    cache = load_teacher_cache(path, "fp", expected_size=2)
    assert cache is not None
    assert set(cache) == {"a", "b"}
    torch.testing.assert_close(cache["a"], torch.tensor([1.0, 2.0, 3.0]))


def test_teacher_cache_rejects_wrong_fingerprint(tmp_path) -> None:
    path = save_teacher_cache(tmp_path / "teacher_train.npz", ["a"], np.zeros((1, 3)), "fp")
    assert load_teacher_cache(path, "other", expected_size=1) is None


def test_teacher_cache_rejects_wrong_size(tmp_path) -> None:
    path = save_teacher_cache(tmp_path / "teacher_train.npz", ["a"], np.zeros((1, 3)), "fp")
    assert load_teacher_cache(path, "fp", expected_size=2) is None


def test_teacher_fingerprint_changes_with_file(tmp_path) -> None:
    path = tmp_path / "teacher.pt"
    path.write_bytes(b"one")
    first = teacher_fingerprint(path)
    path.write_bytes(b"three")
    second = teacher_fingerprint(path)
    assert first != second


def test_rms_normalize_hits_target() -> None:
    waveform = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    normalized = rms_normalize(waveform, 1.0)
    rms = normalized.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(2), atol=1e-5, rtol=1e-5)
