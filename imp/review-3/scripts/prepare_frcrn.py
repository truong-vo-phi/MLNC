"""Prepare pinned FRCRN residual arrays and the matching clean-v2 manifests."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_frcrn(data_dir, cache_dir, remove_archive=False):
    source = json.loads((ROOT / "config/frcrn_sources.json").read_text())
    data_dir, cache_dir = Path(data_dir).resolve(), Path(cache_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sibling = ROOT.parent / "mix_dataset_v2"
    spec = importlib.util.spec_from_file_location("_masked_download", sibling / "download_team_mix_v2.py")
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    manifest_root = data_dir / "mix-dataset-v2"
    files = ["train/manifest.csv", "eval/manifest.csv", "labels.txt"]
    if not all((manifest_root / f).is_file() for f in files):
        tree = json.loads((sibling / "config/hf_dataset_files.json").read_text())
        info = next(x for x in tree if x["path"] == "mix-dataset-v2.zip")
        archive = downloader.download_ranges(source["repo"], source["revision"], info["path"], data_dir, info["size"])
        if sha256(archive) != info["lfs"]["oid"]:
            raise ValueError("Raw metadata archive SHA256 mismatch")
        with ZipFile(archive) as z:
            for name in files:
                member = "mix-dataset-v2/" + name
                target = manifest_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        if remove_archive:
            archive.unlink()
    for split, expected in source["manifest_sha256"].items():
        if sha256(manifest_root / split / "manifest.csv") != expected:
            raise ValueError(f"Wrong clean-v2 {split} manifest")
    if sha256(manifest_root / "labels.txt") != source["labels_sha256"]:
        raise ValueError("Class order differs from the raw baseline")
    if not all((cache_dir / a["file"]).is_file() for a in source["arrays"].values()):
        if shutil.disk_usage(cache_dir).free < source["archive_size"] + 9 * 2**30:
            raise RuntimeError("Need about 17 GiB free for FRCRN ZIP + arrays, plus weights/checkpoints")
        archive = downloader.download_ranges(source["repo"], source["revision"], source["archive"], cache_dir, source["archive_size"])
        if sha256(archive) != source["archive_sha256"]:
            raise ValueError("FRCRN archive SHA256 mismatch")
        with ZipFile(archive) as z:
            for item in source["arrays"].values():
                target = cache_dir / item["file"]
                with z.open(item["file"]) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
                print("Extracted FRCRN", item["file"], flush=True)
        if remove_archive:
            archive.unlink()
    provenance = {"input_kind": "frcrn", "source": source, "files": {}, "manifest_sha256": source["manifest_sha256"]}
    for split, item in source["arrays"].items():
        path = cache_dir / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Unrecognized FRCRN array: {path.name}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(arr.shape) != item["shape"] or arr.dtype != np.float32:
            raise ValueError(f"Invalid shape/dtype for {path.name}")
        with (manifest_root / split / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(arr):
            raise ValueError("FRCRN/manifest row count mismatch")
        for start in range(0, len(arr), 128):
            if not np.isfinite(arr[start:start + 128]).all():
                raise ValueError("Non-finite FRCRN samples")
        stat = path.stat()
        provenance["files"][split] = {"file": path.name, "sha256": item["sha256"], "size": stat.st_size,
                                       "mtime_ns": stat.st_mtime_ns, "shape": list(arr.shape),
                                       "ids_sha256": hashlib.sha256("\n".join(r["sample_id"] for r in rows).encode()).hexdigest()}
        del arr
        print("Verified FRCRN array + clean manifest:", split, len(rows), flush=True)
    target = cache_dir / "frcrn_provenance.json"
    target.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("FRCRN cache ready. No new RMS normalization; no test audio opened.", flush=True)
    return provenance
