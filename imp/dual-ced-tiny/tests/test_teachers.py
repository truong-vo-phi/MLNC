from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

from src.dataset import TARGET_RMS
from src.teachers import compute_teacher_logits


class _WaveDataset(Dataset):
    def __init__(self, n: int = 4, value: float = 0.01) -> None:
        self.wave = torch.full((n, 64000), value)

    def __len__(self) -> int:
        return len(self.wave)

    def __getitem__(self, i: int) -> dict:
        return {"waveform": self.wave[i]}


class _RMSRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, wave: torch.Tensor):
        self.seen.append(wave.detach())
        return SimpleNamespace(logits=torch.zeros(wave.size(0), 3))


def test_compute_teacher_logits_rms_normalizes_before_forward():
    teacher = _RMSRecorder()
    logits = compute_teacher_logits(
        teacher,
        _WaveDataset(),
        batch_size=2,
        device=torch.device("cpu"),
        amp=False,
    )

    assert logits.shape == (4, 3)
    assert len(teacher.seen) == 2
    rms = teacher.seen[0].pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, TARGET_RMS), atol=1.0e-4)


def test_compute_teacher_logits_cache_roundtrip(tmp_path):
    teacher = _RMSRecorder()
    cache = tmp_path / "teacher.npy"
    first = compute_teacher_logits(teacher, _WaveDataset(), 2, torch.device("cpu"), cache_path=cache, amp=False)
    assert cache.is_file()
    calls = len(teacher.seen)

    second = compute_teacher_logits(teacher, _WaveDataset(), 2, torch.device("cpu"), cache_path=cache, amp=False)
    assert len(teacher.seen) == calls  # cache hit: teacher was not called again
    assert (first == second).all()
