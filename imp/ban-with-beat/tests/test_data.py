from __future__ import annotations

import json

import pytest
from ban_beat.data import (
    LabelBalancedSampler,
    check_split_disjoint,
    class_names_of,
    coverage_report,
    dataset_rows,
    label_counts,
    labels_of,
)


class FakeDataset:
    def __init__(self, rows: list[dict[str, str]], class_names: list[str]) -> None:
        self.rows = rows
        self.class_names = class_names


def _row(sample_id: str, snr: float, label: int, noise: str, speech: str) -> dict[str, str]:
    multi_hot = [0] * 3
    multi_hot[label] = 1
    return {
        "sample_id": sample_id,
        "target_snr_db": str(snr),
        "multi_hot_36": json.dumps(multi_hot),
        "noise_ytid": noise,
        "speech_id": speech,
    }


def test_label_balanced_sampler_covers_every_index() -> None:
    labels = [0] * 6 + [1] * 2
    sampler = LabelBalancedSampler(labels, seed=3)
    order = list(sampler)
    assert len(order) == len(labels)
    assert sorted(order) == list(range(len(labels)))
    assert sampler.epoch == 1
    assert sum(index < 6 for index in order[:4]) == 2
    assert sum(index >= 6 for index in order[:4]) == 2


def test_label_balanced_sampler_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LabelBalancedSampler([])


def test_dataset_helpers() -> None:
    dataset = FakeDataset(
        [
            _row("a", -5.0, 0, "n1", "s1"),
            _row("b", 5.0, 1, "n2", "s2"),
            _row("c", 5.0, 1, "n3", "s3"),
        ],
        ["zero", "one", "two"],
    )
    assert dataset_rows(dataset)[0]["sample_id"] == "a"
    assert class_names_of(dataset) == ["zero", "one", "two"]
    assert labels_of(dataset) == [0, 1, 1]
    assert label_counts(dataset) == {0: 1, 1: 2}


def test_split_disjoint_accepts_clean_splits() -> None:
    train = FakeDataset([_row("a", -5.0, 0, "n1", "s1")], ["zero", "one", "two"])
    validation = FakeDataset([_row("b", 5.0, 1, "n2", "s2")], ["zero", "one", "two"])
    overlaps = check_split_disjoint({"train": train, "validation": validation})
    assert all(value == 0 for value in overlaps.values())


def test_split_disjoint_rejects_leakage() -> None:
    train = FakeDataset([_row("a", -5.0, 0, "n1", "s1")], ["zero", "one", "two"])
    validation = FakeDataset([_row("b", 5.0, 1, "n1", "s2")], ["zero", "one", "two"])
    with pytest.raises(ValueError, match="split leakage"):
        check_split_disjoint({"train": train, "validation": validation})


def test_coverage_report() -> None:
    train = FakeDataset(
        [_row("a", -5.0, 0, "n1", "s1"), _row("b", 5.0, 1, "n2", "s2")],
        ["zero", "one", "two"],
    )
    report = coverage_report({"train": train}, (-5.0, 5.0))
    assert report["class_counts"]["train"]["zero"] == 1
    assert report["class_counts"]["train"]["two"] == 0
    assert report["snr_counts"]["train"] == {"-5": 1, "5": 1}
    assert report["missing_classes"]["train"] == ["two"]
