from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from scripts.load_dataset import AUDIOSET_KEY, MIXTURE_KEY, dataset_env_name, resolve_dataset_root

ROOT = Path(__file__).resolve().parents[1]
URL_CONFIG = ROOT / "config" / "url.yaml"
TOKEN_ENV = "KAGGLE_ACCESS_TOKEN"
KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download"
CHUNK_SIZE = 4 * 1024 * 1024

logger = logging.getLogger(__name__)


def resolve_slug(key: str, url_config: Path = URL_CONFIG) -> str:
    with open(url_config, encoding="utf-8") as f:
        slug = yaml.safe_load(f).get(key)
    if not slug:
        raise KeyError(f"'{key}' is missing or empty in {url_config}")
    return slug


def read_token(env_name: str = TOKEN_ENV) -> str:
    token = os.environ.get(env_name)
    if not token:
        raise RuntimeError(f"Set the {env_name} environment variable (never hard-code the token)")
    return token


def download_http(url: str, destination: Path, headers: dict[str, str] | None = None) -> Path:
    if destination.exists():
        logger.info("Already exists: %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with (
            partial.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=destination.name) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                progress.update(len(chunk))
    partial.rename(destination)
    return destination


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid ZIP file: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in tqdm(archive.infolist(), desc=f"Extracting {zip_path.name}", unit="file"):
            archive.extract(member, extract_dir)
    return extract_dir


def download_kaggle_dataset(key: str, root: Path, name: str, url_config: Path = URL_CONFIG) -> Path:
    extract_dir = root / name
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("%s already extracted: %s", name, extract_dir)
        return extract_dir

    url = f"{KAGGLE_API}/{resolve_slug(key, url_config)}"
    headers = {"Authorization": f"Bearer {read_token()}"}
    zip_path = download_http(url, root / f"{name}.zip", headers)
    extract_zip(zip_path, extract_dir)
    logger.info("%s extracted: %s", name, extract_dir)
    return extract_dir


@dataclass(frozen=True)
class DatasetSpec:
    kaggle_key: str
    name: str
    marker: str


DATASETS: dict[str, DatasetSpec] = {
    AUDIOSET_KEY: DatasetSpec("kaggle-audioset", "audioset", "metadata/manifest.csv"),
    MIXTURE_KEY: DatasetSpec("kaggle-mix-dataset", "mix-dataset", "labels.txt"),
}


def find_root(extract_dir: Path, marker: str) -> Path:
    parts = Path(marker).parts
    candidates = sorted(
        (path for path in extract_dir.rglob(parts[-1]) if path.parts[-len(parts) :] == parts),
        key=lambda path: len(path.parts),
    )
    if not candidates:
        raise FileNotFoundError(f"'{marker}' not found under {extract_dir}")
    return candidates[0].parents[len(parts) - 1]


def ensure_dataset(key: str, download_root: Path, url_config: Path = URL_CONFIG) -> Path:
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}', expected one of {sorted(DATASETS)}")
    try:
        return resolve_dataset_root(key, url_config)
    except (KeyError, FileNotFoundError):
        logger.info("%s not found locally, using %s", key, download_root)

    spec = DATASETS[key]
    extract_dir = download_kaggle_dataset(spec.kaggle_key, download_root, spec.name, url_config)
    root = find_root(extract_dir, spec.marker)
    os.environ[dataset_env_name(key)] = str(root)
    return root
