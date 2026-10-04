from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import gcd
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
URL_CONFIG = ROOT / "config" / "url.yaml"
LABELS_FALLBACK = ROOT / "config" / "audioset_40_labels.csv"
AUDIOSET_KEY = "audioset-40l"
MIXTURE_KEY = "mixture-36l"
SAMPLE_RATE = 16000
ALL_SNR = (-5, 0, 5, 10, 15, 20)

logger = logging.getLogger(__name__)


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    snr_db: Tensor
    sample_id: str
    noise: Tensor
    clean: Tensor


def dataset_env_name(key: str) -> str:
    return f"MLNC_{key.upper().replace('-', '_')}"


def resolve_dataset_root(key: str, url_config: Path = URL_CONFIG) -> Path:
    value = os.environ.get(dataset_env_name(key))
    if not value:
        with open(url_config, encoding="utf-8") as f:
            value = yaml.safe_load(f).get(key)
    if not value:
        raise KeyError(f"'{key}' is missing or empty in {url_config}")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fit_length(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) >= num_samples:
        return waveform[:num_samples]
    return np.pad(waveform, (0, num_samples - len(waveform)))


def load_waveform(
    path: Path,
    sample_rate: int,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> Tensor:
    info = sf.info(path)
    start, frames = 0, -1
    if num_samples is not None:
        frames = round(num_samples * info.samplerate / sample_rate)
        slack = max(info.frames - frames, 0)
        start = int(rng.integers(0, slack + 1)) if rng is not None and slack > 0 else 0
    data, source_rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, source_rate // divisor)
    if num_samples is not None:
        waveform = fit_length(waveform, num_samples)
    return torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    speech_power = float(np.mean(speech**2)) + 1e-8
    noise_power = float(np.mean(noise**2)) + 1e-8
    noise = noise * np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixture = speech + noise
    peak = float(np.max(np.abs(mixture)))
    if peak > 1.0:
        mixture, noise = mixture / peak, noise / peak
    return mixture.astype(np.float32), noise.astype(np.float32)


class AudioDataset(Dataset, ABC):
    dataset_key: str

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = None,
    ) -> None:
        self.root = Path(root) if root else resolve_dataset_root(self.dataset_key)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = None if duration is None else int(duration * sample_rate)

    def read(self, path: Path, rng: np.random.Generator | None = None) -> Tensor:
        return load_waveform(path, self.sample_rate, self.num_samples, rng)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> AudioItem: ...


class LabeledAudioDataset(AudioDataset):
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class AudioSet40Dataset(LabeledAudioDataset):
    dataset_key = AUDIOSET_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = 10.0,
        exclude_ytids: set[str] | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        metadata_dir = self.root / "metadata"
        excluded = exclude_ytids or set()

        labels_file = metadata_dir / "40_noise_labels.csv"
        labels = read_csv(labels_file if labels_file.is_file() else LABELS_FALLBACK)
        self.class_names = [row["name"] for row in sorted(labels, key=lambda r: int(r["noise_index"]))]
        self.mid_to_index = {row["mid"]: int(row["noise_index"]) for row in labels}

        rows = [
            row
            for row in read_csv(metadata_dir / "manifest.csv")
            if row["split"] == split and row["status"] == "copied" and row["ytid"] not in excluded
        ]
        samples = [(self.root / "audio" / f"{row['ytid']}.wav", self._multi_hot(row["label_mids"])) for row in rows]
        self.samples = [(path, target) for path, target in samples if path.is_file()]
        if len(self.samples) < len(rows):
            logger.warning("%s: skipped %d rows without audio file", split, len(rows) - len(self.samples))

    def _multi_hot(self, label_mids: str) -> Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for mid in label_mids.split(","):
            index = self.mid_to_index.get(mid.strip())
            if index is not None:
                target[index] = 1.0
        return target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AudioItem:
        path, target = self.samples[index]
        return {"waveform": self.read(path), "target": target}


class Mixture36Dataset(LabeledAudioDataset):
    dataset_key = MIXTURE_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = 4.0,
        load_noise: bool = False,
        load_clean: bool = False,
        snr_db: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        self.load_noise = load_noise
        self.load_clean = load_clean

        with open(self.root / "labels.txt", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        rows = read_csv(self.root / split / "manifest.csv")
        if snr_db is not None:
            allowed = set(snr_db)
            rows = [row for row in rows if float(row["target_snr_db"]) in allowed]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        item: AudioItem = {
            "waveform": self.read(self.root / row["mixture_path"]),
            "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
            "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }
        if self.load_noise:
            item["noise"] = self.read(self.root / row["noise_path"])
        if self.load_clean:
            item["clean"] = self.read(self.root / row["clean_path"])
        return item


class DynamicMixDataset(AudioDataset):
    dataset_key = MIXTURE_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        noise_dataset: AudioSet40Dataset | None = None,
        snr_levels: Sequence[float] = ALL_SNR,
        sample_rate: int = SAMPLE_RATE,
        duration: float = 2.0,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        rows = read_csv(self.root / split / "manifest.csv")
        self.speech_paths = sorted({self.root / row["clean_path"] for row in rows})
        self.noise_dataset = noise_dataset or AudioSet40Dataset(split=split, sample_rate=sample_rate, duration=None)
        self.snr_levels = tuple(snr_levels)
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return len(self.speech_paths)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + index if self.deterministic else None)

    def __getitem__(self, index: int) -> AudioItem:
        rng = self._rng(index)
        noise_path, _ = self.noise_dataset.samples[int(rng.integers(0, len(self.noise_dataset)))]
        speech = self.read(self.speech_paths[index], rng).numpy()
        noise = self.read(noise_path, rng).numpy()
        snr_db = float(rng.choice(self.snr_levels))
        mixture, noise = mix_at_snr(speech, noise, snr_db)
        return {
            "waveform": torch.from_numpy(mixture),
            "noise": torch.from_numpy(noise),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


def build_dataset(name: str, split: str, **kwargs: Any) -> AudioDataset:
    builders: dict[str, type[AudioDataset]] = {AUDIOSET_KEY: AudioSet40Dataset, MIXTURE_KEY: Mixture36Dataset}
    if name not in builders:
        raise ValueError(f"Unknown dataset '{name}', expected one of {sorted(builders)}")
    return builders[name](split=split, **kwargs)


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def summarize(name: str, splits: tuple[str, ...]) -> None:
    for split in splits:
        dataset = build_dataset(name, split)
        shapes = {k: tuple(v.shape) for k, v in dataset[0].items() if isinstance(v, Tensor)}
        logger.info("%s/%s: %d samples, %d classes, %s", name, split, len(dataset), dataset.num_classes, shapes)


def read_noise_ytids(root: str | Path | None = None, splits: Sequence[str] = ("validation", "test")) -> set[str]:
    root = Path(root) if root else resolve_dataset_root(MIXTURE_KEY)
    return {row["noise_ytid"] for split in splits for row in read_csv(root / split / "manifest.csv")}


def subsample(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, max_samples).round().astype(int)
    return Subset(dataset, indices.tolist())
