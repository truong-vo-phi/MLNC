from __future__ import annotations

import csv
import json
import logging
import os
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
NUM_SAMPLES = 64000  # 4 seconds at 16 kHz
NUM_CLASSES = 36


def read_audio_16k_mono(
    path: Path,
    target_samples: int = NUM_SAMPLES,
    target_sr: int = SAMPLE_RATE,
) -> Tensor:
    """Read any WAV file using scipy, downmix to mono, resample to 16kHz, fit to target_samples."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        waveform = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        waveform = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        waveform = (data.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = data.astype(np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    if sr != target_sr:
        g = gcd(target_sr, sr)
        waveform = resample_poly(waveform, target_sr // g, sr // g)

    if len(waveform) >= target_samples:
        waveform = waveform[:target_samples]
    else:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)))

    return torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))


def load_36_class_names(path: str | Path | None = None) -> list[str]:
    """Load the 36 class names from labels.txt or partition_labels_36.json."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            if p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                return [item["name"] for item in sorted(data, key=lambda x: int(x.get("label_index", 0)))]
            with open(p, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

    # Search common fallback paths
    candidates = [
        Path("dataset/mix-dataset-v2/mix-dataset-v2/labels.txt"),
        Path("datasets/mix-dataset/labels.txt"),
        Path("partition_labels_36.json"),
        Path(__file__).resolve().parents[3] / "partition_labels_36.json",
    ]
    for c in candidates:
        if c.is_file():
            if c.suffix == ".json":
                data = json.loads(c.read_text(encoding="utf-8"))
                return [item["name"] for item in sorted(data, key=lambda x: int(x.get("label_index", 0)))]
            with open(c, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

    # Built-in standard fallback list of the 36 AudioSet noise classes
    return [
        "Acoustic guitar", "Applause", "Bark", "Burping, eructation", "Bus", "Chime",
        "Clarinet", "Computer keyboard", "Cough", "Cowbell", "Cricket", "Cupboard open or close",
        "Dishes, pots, and pans", "Door", "Drawer open or close", "Electric guitar", "Fart",
        "Finger snapping", "Giggle", "Glockenspiel", "Gong", "Harmonica", "Hi-hat",
        "Knock", "Laughter", "Microwave oven", "Motorcycle", "Pant", "Printer",
        "Rain", "Rattle", "Scissors", "Sigh", "Snicker", "Sneeze", "Violin, fiddle"
    ]


class AudioSet36Dataset(Dataset):
    """AudioSet dataset filtered for the 36 classes (Teacher domain: clean acoustic sound events)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        class_names: list[str] | None = None,
        duration: float = 4.0,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.class_names = class_names or load_36_class_names()
        self.name_to_idx = {name.strip(): i for i, name in enumerate(self.class_names)}
        self.target_samples = int(duration * SAMPLE_RATE)

        manifest_path = self.root / "metadata" / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"AudioSet manifest not found at: {manifest_path}")

        audio_dir = self.root / "audio"
        self.samples: list[tuple[Path, Tensor, str]] = []

        with open(manifest_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {k.strip('\ufeff"').strip(): v for k, v in raw_row.items() if k is not None}
                if row.get("split") != split:
                    continue
                if row.get("status") not in (None, "", "copied"):
                    continue

                ytid = str(row.get("ytid", "")).strip('\"')
                wav_file = audio_dir / f"{ytid}.wav"
                if not wav_file.is_file():
                    continue

                # Match labels
                label_names = [n.strip().strip('"') for n in str(row.get("label_names", "")).split(",")]
                target = torch.zeros(len(self.class_names), dtype=torch.float32)
                matched = False
                for name in label_names:
                    idx = self.name_to_idx.get(name)
                    if idx is not None:
                        target[idx] = 1.0
                        matched = True

                if matched:
                    self.samples.append((wav_file, target, ytid))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break

        logger.info("AudioSet36Dataset [%s]: Loaded %d clips matching 36 classes", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, target, ytid = self.samples[index]
        waveform = read_audio_16k_mono(path, target_samples=self.target_samples)
        return {
            "waveform": waveform,
            "target": target,
            "sample_id": ytid,
        }


def find_mix_manifest(cache_dir: str | Path, split: str) -> Path | None:
    """Locate the mix-dataset manifest of `split` ('train' | 'validation') near the cache directory.

    Walks the parent of `cache_dir` (the zip extracted on Windows may contain directory names with a
    literal backslash, so fixed paths are not reliable) and picks the manifest whose `split` column matches.
    """
    cache_dir = Path(cache_dir)
    roots = [cache_dir.parent, cache_dir.parent.parent, Path("dataset"), Path("datasets")]
    skip_dirs = {"output", ".git", "__pycache__", "audio", "src", "scripts"}
    seen_roots: set[str] = set()
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir() or str(root.resolve()) in seen_roots:
            continue
        seen_roots.add(str(root.resolve()))
        base_depth = len(root.resolve().parts)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            if len(Path(dirpath).resolve().parts) - base_depth > 5:
                dirnames[:] = []
            if "manifest.csv" not in filenames:
                continue
            path = Path(dirpath) / "manifest.csv"
            key = str(path.resolve())
            if key in seen or "audioset" in key.lower():
                continue
            seen.add(key)
            try:
                with open(path, encoding="utf-8-sig", newline="") as f:
                    first = next(csv.DictReader(f), None)
            except OSError:
                continue
            if first and "multi_hot_36" in first and first.get("split") == split:
                return path
    return None


class CacheNhatDataset(Dataset):
    """Dataset loading pre-extracted noise estimation n_hat from FRCRN masking (Student domain)."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        manifest_path: str | Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        npy_path = self.cache_dir / f"{split}.npy"
        done_path = self.cache_dir / f"{split}.done"

        if not npy_path.is_file():
            raise FileNotFoundError(f"cache-nhat file not found: {npy_path}")

        self.array = np.load(npy_path, mmap_mode="r")
        total_rows = len(self.array)

        # Find manifest (holds multi_hot_36 labels and noise_ytid for every cached row)
        self.rows: list[dict[str, Any]] = []
        resolved_manifest = None
        if manifest_path is not None and Path(manifest_path).is_file():
            resolved_manifest = Path(manifest_path)
        else:
            resolved_manifest = find_mix_manifest(self.cache_dir, split)

        if resolved_manifest is None:
            raise FileNotFoundError(
                f"mix manifest for split '{split}' not found near {self.cache_dir.parent}; "
                "refusing to continue because every label would silently become zero"
            )
        with open(resolved_manifest, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        logger.info("CacheNhatDataset [%s]: manifest %s (%d rows)", split, resolved_manifest, len(self.rows))

        if self.rows and len(self.rows) != total_rows:
            logger.warning(
                "CacheNhatDataset [%s]: manifest row count (%d) != npy count (%d)",
                split, len(self.rows), total_rows
            )

        self.length = min(total_rows, len(self.rows)) if self.rows else total_rows
        if max_samples is not None:
            self.length = min(self.length, max_samples)

        logger.info("CacheNhatDataset [%s]: Loaded %d clips from %s", split, self.length, npy_path)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        waveform = torch.from_numpy(np.array(self.array[index], dtype=np.float32))
        item: dict[str, Any] = {"waveform": waveform}

        row = self.rows[index]
        item["target"] = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
        item["snr_db"] = torch.tensor(float(row.get("target_snr_db", 0.0)), dtype=torch.float32)
        item["sample_id"] = row.get("sample_id", str(index))
        item["noise_ytid"] = str(row.get("noise_ytid", "")).strip().strip('"')
        item["index"] = torch.tensor(index, dtype=torch.long)

        return item


class PairedStudentTeacherDataset(Dataset):
    """Pairs each cache-nhat row (Student input) with its clean AudioSet noise clip (Teacher input).

    The pairing key is the `noise_ytid` column of the mix manifest, which names the AudioSet
    clip `<audioset_dir>/audio/<ytid>.wav` that the noise of the mixture was taken from.
    Items: `waveform` (Student, FRCRN-masked mix), `teacher_waveform` (Teacher, clean AudioSet),
    `target` (36-way multi-hot), plus ids.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        audioset_dir: str | Path,
        split: str = "train",
        duration: float = 4.0,
        max_samples: int | None = None,
    ) -> None:
        self.base = CacheNhatDataset(cache_dir=cache_dir, split=split, max_samples=max_samples)
        audio_dir = Path(audioset_dir) / "audio"
        self.target_samples = int(duration * SAMPLE_RATE)

        self.indices: list[int] = []
        missing = 0
        for index in range(len(self.base)):
            ytid = str(self.base.rows[index].get("noise_ytid", "")).strip().strip('"')
            if ytid and (audio_dir / f"{ytid}.wav").is_file():
                self.indices.append(index)
            else:
                missing += 1
        if not self.indices:
            raise FileNotFoundError(f"no AudioSet clip in {audio_dir} matches any noise_ytid of split '{split}'")
        if missing:
            logger.warning("PairedStudentTeacherDataset [%s]: dropped %d rows without AudioSet clip", split, missing)

        # Preload each unique clean clip once (few thousand clips) so epochs are not I/O bound.
        self.clean: dict[str, np.ndarray] = {}
        for index in self.indices:
            ytid = str(self.base.rows[index]["noise_ytid"]).strip().strip('"')
            if ytid not in self.clean:
                clip = read_audio_16k_mono(audio_dir / f"{ytid}.wav", target_samples=self.target_samples)
                self.clean[ytid] = clip.numpy()
        logger.info(
            "PairedStudentTeacherDataset [%s]: %d pairs, %d unique clean clips",
            split, len(self.indices), len(self.clean),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[self.indices[index]])
        item["index"] = torch.tensor(index, dtype=torch.long)
        item["teacher_waveform"] = torch.from_numpy(self.clean[item["noise_ytid"]].copy())
        return item


def build_student_teacher_loaders(
    cache_nhat_dir: str | Path,
    audioset_dir: str | Path,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, DataLoader]:
    """Build paired data loaders for Student (cache-nhat) and Teacher (AudioSet36)."""
    # 1. Student datasets (cache-nhat)
    student_train_ds = CacheNhatDataset(
        cache_dir=cache_nhat_dir,
        split="train",
        max_samples=max_train_samples,
    )
    student_val_ds = CacheNhatDataset(
        cache_dir=cache_nhat_dir,
        split="validation",
        max_samples=max_eval_samples,
    )

    # 2. Teacher datasets (AudioSet 36)
    teacher_train_ds = AudioSet36Dataset(
        root=audioset_dir,
        split="train",
        max_samples=max_train_samples,
    )
    teacher_val_ds = AudioSet36Dataset(
        root=audioset_dir,
        split="validation",
        max_samples=max_eval_samples,
    )

    loaders = {
        "student_train": DataLoader(
            student_train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
        "student_val": DataLoader(
            student_val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "teacher_train": DataLoader(
            teacher_train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
        "teacher_val": DataLoader(
            teacher_val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return loaders
