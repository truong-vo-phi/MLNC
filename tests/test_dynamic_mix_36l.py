import numpy as np
import torch

from scripts.dynamic_mix_36l import dropped_label_names, filter_noise_36l


class FakeNoise:
    class_names = ["a", "b", "c", "d"]

    def __init__(self):
        multi_hot = lambda *idx: torch.tensor([1.0 if i in idx else 0.0 for i in range(4)])  # noqa: E731
        self.samples = [("p0", multi_hot(0)), ("p1", multi_hot(1)), ("p2", multi_hot(0, 3)), ("p3", multi_hot(2))]


def test_dropped_labels_are_the_four_missing_from_36l():
    from scripts.load_dataset import LABELS_FALLBACK, read_csv

    names = [row["name"] for row in read_csv(LABELS_FALLBACK)]
    dropped = dropped_label_names(names[:36])
    assert dropped == names[36:] and len(dropped) == 4
    assert dropped_label_names(names) == []


def test_filter_removes_clips_with_any_dropped_label():
    noise = FakeNoise()
    removed = filter_noise_36l(noise, ["b", "d"])
    assert removed == 2
    assert [path for path, _ in noise.samples] == ["p0", "p3"]
    assert np.isfinite(removed)
