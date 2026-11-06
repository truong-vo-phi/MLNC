"""Verify ablation data and retrieve official CED-Tiny initialization weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/ablations.yaml").read_text(encoding="utf-8"))
    defaults = cfg["defaults"]
    train_path = Path(defaults["train_array"])
    val_path = Path(defaults["validation_array"])
    manifest_root = Path(defaults["manifest_root"])
    required = [train_path, val_path, manifest_root / "train/manifest.csv", manifest_root / "eval/manifest.csv"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
        print(f"FOUND {path} ({path.stat().st_size:,} bytes)", flush=True)
    for path, expected in ((train_path, (30240, 64000)), (val_path, (3064, 64000))):
        array = np.load(path, mmap_mode="r")
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"Invalid cache {path}: shape={array.shape}, dtype={array.dtype}; expected {expected} float32")
        print(f"CACHE OK {path}: {array.shape} {array.dtype}", flush=True)

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub and safetensors in the notebook environment before Run all.") from exc
    weight_path = Path(hf_hub_download("mispeech/ced-tiny", "model.safetensors", local_dir=ROOT / "pretrained"))
    state = load_file(str(weight_path), device="cpu")
    if not any(key.startswith("encoder.blocks.") for key in state):
        raise RuntimeError("Downloaded CED checkpoint has no encoder block tensors.")
    teacher_path = Path(hf_hub_download(
        "phitruong0606/mix-dataset-v2", "beats_teacher_fold0_weights.pt",
        repo_type="dataset", local_dir=ROOT / "pretrained",
    ))
    teacher_blob = __import__("torch").load(teacher_path, map_location="cpu", weights_only=False)
    teacher_state = teacher_blob.get("state_dict", teacher_blob)
    if not isinstance(teacher_state, dict) or not teacher_state:
        raise RuntimeError("BEATs teacher checkpoint has no state dictionary.")
    receipt = {
        "repo": "mispeech/ced-tiny",
        "file": str(weight_path),
        "sha256": sha256(weight_path),
        "tensors": len(state),
        "bytes": weight_path.stat().st_size,
        "teacher_file": str(teacher_path),
        "teacher_sha256": sha256(teacher_path),
        "cache_manifest_alignment": "assumed row order; cache archive contains no sample IDs",
    }
    out = ROOT / "logs/ablations/preparation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("PREPARATION COMPLETE", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
