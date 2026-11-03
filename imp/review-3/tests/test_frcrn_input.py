"""Guard manifest alignment and cache tampering without GPU or large datasets."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.dataset import VerifiedFRCRNDataset


@pytest.fixture
def masked_fixture(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    (root / "eval").mkdir(parents=True)
    manifest = root / "eval/manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "multi_hot_36", "target_snr_db"])
        writer.writeheader()
        for i in range(2):
            target = [0] * 36
            target[i + 1] = 1
            writer.writerow({"sample_id": f"validation_{i}", "multi_hot_36": json.dumps(target), "target_snr_db": [-5, 20][i]})
    cache = tmp_path / "cache"
    cache.mkdir()
    array = np.stack([np.full(64000, .01, dtype=np.float32), np.full(64000, .2, dtype=np.float32)])
    np.save(cache / "validation.npy", array)
    file = cache / "validation.npy"
    info = {"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "shape": [2, 64000]}
    source = {"arrays": {"eval": info}, "manifest_sha256": {"eval": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    stat = file.stat()
    entry = {**info, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "ids_sha256": hashlib.sha256(b"validation_0\nvalidation_1").hexdigest()}
    (cache / "frcrn_provenance.json").write_text(json.dumps({"input_kind": "frcrn", "files": {"eval": entry}}))
    original = Path.read_text
    def read_text(path, *args, **kwargs):
        if path.name == "frcrn_sources.json":
            return json.dumps(source)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    return root, cache, array


def test_masked_waveform_labels_and_worker_pickle(masked_fixture):
    root, cache, array = masked_fixture
    ds = VerifiedFRCRNDataset(root, cache, split="eval")
    assert np.array_equal(ds[1]["waveform"].numpy(), array[1])
    assert ds[1]["target_idx"].item() == 2
    assert ds[1]["sample_id"] == "validation_1"
    assert ds[1]["snr_db"].item() == 20
    assert ds.__getstate__()["_waveform_array"] is None


def test_modified_array_rejected(masked_fixture):
    root, cache, _ = masked_fixture
    file = cache / "validation.npy"
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    with pytest.raises(ValueError, match="changed after verification"):
        VerifiedFRCRNDataset(root, cache, split="eval")


def test_reordered_manifest_rejected(masked_fixture):
    root, cache, _ = masked_fixture
    manifest = root / "eval/manifest.csv"
    lines = manifest.read_text().splitlines(keepends=True)
    manifest.write_text(lines[0] + lines[2] + lines[1])
    with pytest.raises(ValueError, match="manifest mismatch"):
        VerifiedFRCRNDataset(root, cache, split="eval")


def test_wrong_cache_source_rejected(masked_fixture):
    root, cache, _ = masked_fixture
    path = cache / "frcrn_provenance.json"
    proof = json.loads(path.read_text())
    proof["files"]["eval"]["file"] = "train.npy"
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="pinned team cache"):
        VerifiedFRCRNDataset(root, cache, split="eval")
