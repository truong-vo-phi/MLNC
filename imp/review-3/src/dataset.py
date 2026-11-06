"""
Dataset Loader for Mixture Audio across Multiple SNR levels.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence, TypedDict

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
TARGET_RMS = 0.07925
RMS_EPS = 1.0e-8


def rms_normalize(waveform: Tensor, target_rms: float = TARGET_RMS, eps: float = RMS_EPS) -> Tensor:
    """
    Standardize the energy level of the waveform to target_rms (default 0.07925).
    Formula: x_norm = x * (target_rms / (sqrt(mean(x^2)) + eps))
    Works for both 1D (num_samples,) and 2D (B, num_samples) tensors.
    """
    if waveform.dim() == 1:
        rms = torch.sqrt(torch.mean(waveform ** 2) + eps)
        return waveform * (target_rms / rms)
    rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True) + eps)
    return waveform * (target_rms / rms)


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    target_idx: Tensor
    snr_db: Tensor
    sample_id: str


def read_csv_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class MixtureDataset(Dataset):
    """
    Dataset loader for mixture audio with background noise under speech across SNRs.
    Loads from real disk manifests (train/manifest.csv, validation/manifest.csv).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_classes: int = 36,
        snr_levels: Sequence[float] = SNR_LEVELS,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.num_classes = num_classes
        self.snr_levels = tuple(snr_levels)

        # Locate manifest
        manifest_dir = "eval" if split in ("eval", "val", "validation") else split
        manifest_path = self.root / manifest_dir / "manifest.csv"
        if not manifest_path.is_file():
            manifest_path = self.root / split / "manifest.csv"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        rows = read_csv_manifest(manifest_path)
        if self.snr_levels:
            allowed = set(self.snr_levels)
            rows = [r for r in rows if float(r.get("target_snr_db", r.get("achieved_snr_db", 0.0))) in allowed]

        if max_samples and max_samples < len(rows):
            rows = rows[:max_samples]

        self.rows = rows

        # Load class names if present
        labels_path = self.root / "labels.txt"
        if labels_path.is_file():
            with open(labels_path, encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            self.class_names = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return len(self.rows)

    def _read_audio(self, path: Path) -> Tensor:
        if not path.is_file():
            path = self.root / path
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = data[:, 0]  # Mono
        if sr != self.sample_rate:
            # Simple linear resample if rate differs
            pass
        wave_t = torch.from_numpy(waveform)
        if len(wave_t) < self.num_samples:
            wave_t = torch.nn.functional.pad(wave_t, (0, self.num_samples - len(wave_t)))
        else:
            wave_t = wave_t[: self.num_samples]
        return wave_t

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        audio_path = Path(row.get("mixture_path", row.get("path", "")))
        waveform = self._read_audio(audio_path)

        if "multi_hot_36" in row:
            target_vec = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
            target_idx = target_vec.argmax(dim=-1)
        elif "label" in row:
            target_idx = torch.tensor(int(row["label"]), dtype=torch.long)
            target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            target_vec[target_idx] = 1.0
        else:
            target_idx = torch.tensor(0, dtype=torch.long)
            target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            target_vec[0] = 1.0

        snr = float(row.get("target_snr_db", row.get("achieved_snr_db", 0.0)))
        return {
            "waveform": waveform,
            "target": target_vec,
            "target_idx": target_idx,
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "sample_id": row.get("sample_id", f"sample_{index}"),
        }


class SyntheticMixtureDataset(Dataset):
    """
    Synthetic dataset for smoke testing, offline execution, and unit verification.
    Generates synthetic multi-tone audio with diverse SNR levels and target classes.
    """

    def __init__(
        self,
        num_samples_total: int = 120,
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_classes: int = 36,
        snr_levels: Sequence[float] = SNR_LEVELS,
        seed: int = 42,
    ) -> None:
        self.num_samples_total = num_samples_total
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.num_classes = num_classes
        self.snr_levels = tuple(snr_levels)
        self.seed = seed
        self.class_names = [f"synthetic_class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return self.num_samples_total

    def __getitem__(self, index: int) -> AudioItem:
        rng = np.random.default_rng(self.seed + index)
        cls_idx = index % self.num_classes
        snr_val = self.snr_levels[index % len(self.snr_levels)]

        # Generate synthetic mixture waveform: sine wave (speech) + Gaussian noise
        t = np.linspace(0, 4.0, self.num_samples, endpoint=False)
        base_freq = 200.0 + cls_idx * 25.0
        signal = 0.5 * np.sin(2 * np.pi * base_freq * t).astype(np.float32)
        noise = rng.normal(0, 0.1, size=self.num_samples).astype(np.float32)

        # Apply SNR scaling
        p_sig = np.mean(signal**2) + 1e-8
        p_noise = np.mean(noise**2) + 1e-8
        target_ratio = 10.0 ** (snr_val / 10.0)
        noise = noise * np.sqrt(p_sig / (target_ratio * p_noise))
        mix = (signal + noise).astype(np.float32)

        target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        target_vec[cls_idx] = 1.0

        return {
            "waveform": torch.from_numpy(mix),
            "target": target_vec,
            "target_idx": torch.tensor(cls_idx, dtype=torch.long),
            "snr_db": torch.tensor(snr_val, dtype=torch.float32),
            "sample_id": f"synth_{index:04d}",
        }


class MaskedNoiseCacheDataset(Dataset):
    """
    High-performance zero-copy memory-mapped dataset connecting FRCRN-separated
    noise cache (.npy) with mixture manifest ground truth and SNR levels.

    Args:
        npy_path: Direct path to .npy file or directory containing <split>.npy
        manifest_root: Dataset root (containing manifest.csv or split/manifest.csv)
        split: 'train', 'validation', or 'test'
        sample_rate: Audio sampling rate (default: 16000)
        duration: Audio duration in seconds (default: 4.0)
        num_classes: Number of acoustic noise classes (default: 36)
        apply_rms_norm: Whether to apply energy standardization via rms_normalize (target_rms=0.07925)
        target_rms: Target RMS amplitude (default: 0.07925)
        snr_levels: Sequence of SNR levels to filter or include
        max_samples: Maximum number of samples to load (optional)
    """

    def __init__(
        self,
        npy_path: str | Path,
        manifest_root: str | Path | None = None,
        split: str = "validation",
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_classes: int = 36,
        apply_rms_norm: bool = True,
        target_rms: float = TARGET_RMS,
        snr_levels: Sequence[float] = SNR_LEVELS,
        max_samples: int | None = None,
    ) -> None:
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.num_classes = num_classes
        self.apply_rms_norm = apply_rms_norm
        self.target_rms = target_rms
        self.snr_levels = tuple(snr_levels)

        # 1. Resolve .npy file
        npy_p = Path(npy_path)
        if npy_p.is_dir():
            candidates = [
                npy_p / f"{split}.npy",
                npy_p / f"{split}_noise.npy",
                npy_p / ("eval.npy" if split in ("val", "validation", "eval") else f"{split}.npy"),
            ]
            found = False
            for cand in candidates:
                if cand.is_file():
                    npy_p = cand
                    found = True
                    break
            if not found and split == "test":
                for cand in [npy_p / "validation.npy", npy_p / "eval.npy", npy_p / "validation_noise.npy"]:
                    if cand.is_file():
                        npy_p = cand
                        found = True
                        split = "validation"
                        break
            if not found:
                all_npys = sorted(npy_p.glob("*.npy"))
                if all_npys:
                    npy_p = all_npys[0]
                else:
                    raise FileNotFoundError(f"No .npy cache file found in directory {npy_path}")
        elif not npy_p.is_file():
            raise FileNotFoundError(f"Noise cache .npy file not found: {npy_p}")

        self.npy_file = npy_p
        self._array = np.load(self.npy_file, mmap_mode="r")

        # 2. Resolve manifest for labels and SNR levels
        self.rows: list[dict[str, str]] = []
        manifest_candidates = []
        if manifest_root:
            m_root = Path(manifest_root)
            manifest_candidates.extend([
                m_root / ("eval" if split in ("val", "validation", "eval") else split) / "manifest.csv",
                m_root / split / "manifest.csv",
                m_root / "manifest.csv",
            ])
        manifest_candidates.extend([
            self.npy_file.parent / f"{split}_manifest.csv",
            self.npy_file.parent / "manifest.csv",
        ])

        for cand in manifest_candidates:
            if cand.is_file():
                try:
                    self.rows = read_csv_manifest(cand)
                    break
                except Exception as e:
                    logger.debug("Failed reading manifest %s: %s", cand, e)

        # 3. Map indices ensuring alignment between .npy row and manifest row
        max_available = len(self._array)
        if self.rows:
            max_available = min(max_available, len(self.rows))

        valid_indices = list(range(max_available))
        if self.rows and self.snr_levels:
            allowed = set(self.snr_levels)
            valid_indices = [
                i
                for i in valid_indices
                if float(self.rows[i].get("target_snr_db", self.rows[i].get("achieved_snr_db", 0.0))) in allowed
            ]

        if max_samples and max_samples < len(valid_indices):
            valid_indices = valid_indices[:max_samples]

        self.indices = valid_indices

        # 4. Load class names
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        if manifest_root:
            labels_path = Path(manifest_root) / "labels.txt"
            if labels_path.is_file():
                try:
                    with open(labels_path, encoding="utf-8") as f:
                        names = [line.strip() for line in f if line.strip()]
                    if len(names) == num_classes:
                        self.class_names = names
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> AudioItem:
        real_idx = self.indices[index]
        raw_np = np.array(self._array[real_idx], dtype=np.float32)
        wave_t = torch.from_numpy(raw_np)

        if wave_t.ndim > 1:
            wave_t = wave_t.squeeze()

        if len(wave_t) < self.num_samples:
            wave_t = torch.nn.functional.pad(wave_t, (0, self.num_samples - len(wave_t)))
        elif len(wave_t) > self.num_samples:
            wave_t = wave_t[: self.num_samples]

        if self.apply_rms_norm:
            wave_t = rms_normalize(wave_t, target_rms=self.target_rms)

        if self.rows and real_idx < len(self.rows):
            row = self.rows[real_idx]
            if "multi_hot_36" in row:
                target_vec = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
                target_idx = target_vec.argmax(dim=-1)
            elif "label" in row:
                target_idx = torch.tensor(int(row["label"]), dtype=torch.long)
                target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
                target_vec[target_idx] = 1.0
            else:
                target_idx = torch.tensor(0, dtype=torch.long)
                target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
                target_vec[0] = 1.0
            snr = float(row.get("target_snr_db", row.get("achieved_snr_db", 0.0)))
            sample_id = row.get("sample_id", f"sample_{real_idx:06d}")
        else:
            cls_idx = real_idx % self.num_classes
            target_idx = torch.tensor(cls_idx, dtype=torch.long)
            target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            target_vec[cls_idx] = 1.0
            snr = self.snr_levels[real_idx % len(self.snr_levels)] if self.snr_levels else 0.0
            sample_id = f"sample_{real_idx:06d}"

        return {
            "waveform": wave_t,
            "target": target_vec,
            "target_idx": target_idx,
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "sample_id": sample_id,
        }

    def close(self) -> None:
        """Explicitly release the memory-mapped file handle on Windows."""
        if hasattr(self, "_array") and self._array is not None:
            if hasattr(self._array, "_mmap") and self._array._mmap is not None:
                try:
                    self._array._mmap.close()
                except Exception:
                    pass
            self._array = None

    def __enter__(self) -> MaskedNoiseCacheDataset:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class SyntheticMaskedNoiseCacheDataset(Dataset):
    """
    Synthetic dataset simulating FRCRN-masked noise loaded from memory for smoke testing & unit tests.
    """

    def __init__(
        self,
        num_samples_total: int = 60,
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_classes: int = 36,
        snr_levels: Sequence[float] = SNR_LEVELS,
        apply_rms_norm: bool = True,
        target_rms: float = TARGET_RMS,
        seed: int = 42,
    ) -> None:
        self.num_samples_total = num_samples_total
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.num_classes = num_classes
        self.snr_levels = tuple(snr_levels)
        self.apply_rms_norm = apply_rms_norm
        self.target_rms = target_rms
        self.seed = seed
        self.class_names = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return self.num_samples_total

    def __getitem__(self, index: int) -> AudioItem:
        rng = np.random.default_rng(self.seed + index)
        cls_idx = index % self.num_classes
        snr_val = self.snr_levels[index % len(self.snr_levels)]

        # Generate synthetic isolated noise waveform
        noise = rng.normal(0, 0.05, size=self.num_samples).astype(np.float32)
        wave_t = torch.from_numpy(noise)

        if self.apply_rms_norm:
            wave_t = rms_normalize(wave_t, target_rms=self.target_rms)

        target_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        target_vec[cls_idx] = 1.0

        return {
            "waveform": wave_t,
            "target": target_vec,
            "target_idx": torch.tensor(cls_idx, dtype=torch.long),
            "snr_db": torch.tensor(snr_val, dtype=torch.float32),
            "sample_id": f"masked_synth_{index:04d}",
        }


def get_dataset(
    root: str | Path | None = None,
    split: str = "train",
    num_classes: int = 36,
    sample_rate: int = 16000,
    duration: float = 4.0,
    snr_levels: Sequence[float] = SNR_LEVELS,
    max_samples: int | None = None,
    noise_cache_path: str | Path | None = None,
    apply_rms_norm: bool = True,
    target_rms: float = TARGET_RMS,
    allow_synthetic_fallback: bool = True,
) -> Dataset:
    """
    Unified dataset factory:
    1. If noise_cache_path is provided, loads MaskedNoiseCacheDataset (.npy).
    2. Else if root directory exists with audio/manifests, loads MixtureDataset.
    3. Otherwise falls back to SyntheticMixtureDataset or SyntheticMaskedNoiseCacheDataset.
    """
    if noise_cache_path is not None:
        try:
            return MaskedNoiseCacheDataset(
                npy_path=noise_cache_path,
                manifest_root=root,
                split=split,
                sample_rate=sample_rate,
                duration=duration,
                num_classes=num_classes,
                apply_rms_norm=apply_rms_norm,
                target_rms=target_rms,
                snr_levels=snr_levels,
                max_samples=max_samples,
            )
        except Exception as e:
            if not allow_synthetic_fallback:
                raise e
            logger.warning("Could not load MaskedNoiseCacheDataset (%s), falling back to SyntheticMaskedNoiseCacheDataset", e)
            count = 120 if split == "train" else 60
            if max_samples:
                count = min(count, max_samples)
            return SyntheticMaskedNoiseCacheDataset(
                num_samples_total=count,
                sample_rate=sample_rate,
                duration=duration,
                num_classes=num_classes,
                snr_levels=snr_levels,
                apply_rms_norm=apply_rms_norm,
                target_rms=target_rms,
                seed=42 if split == "train" else 1337,
            )

    if root is not None and Path(root).is_dir():
        try:
            return MixtureDataset(
                root=root,
                split=split,
                sample_rate=sample_rate,
                duration=duration,
                num_classes=num_classes,
                snr_levels=snr_levels,
                max_samples=max_samples,
            )
        except Exception as e:
            if not allow_synthetic_fallback:
                raise e
            logger.warning("Could not load real dataset (%s), falling back to SyntheticMixtureDataset", e)

    count = 120 if split == "train" else 60
    if max_samples:
        count = min(count, max_samples)
    return SyntheticMixtureDataset(
        num_samples_total=count,
        sample_rate=sample_rate,
        duration=duration,
        num_classes=num_classes,
        snr_levels=snr_levels,
        seed=42 if split == "train" else 1337,
    )


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

