from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.drive_sync import BEST_FILENAME, DRIVE_FRCRN_DIR  # noqa: E402
from src.core import BaseModel, LoadReport  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402

URL_CONFIG = ROOT / "config" / "url.yaml"
PRETRAINED_DIR = ROOT / "pretrained"
# FRCRN/best.pt trên Drive (beat_v1); tách khỏi frcrn_best.pt (bản VLSP dùng cho beat-v2)
FRCRN_BEST_PATH = PRETRAINED_DIR / "frcrn_beat_v1_best.pt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainedSpec:
    model_cls: type[BaseModel]
    url_key: str
    config_path: Path


PRETRAINED: dict[str, PretrainedSpec] = {
    "beats": PretrainedSpec(BEATClassifier, "beats-pretrain", ROOT / "config" / "beat.yaml"),
    "frcrn": PretrainedSpec(FRCRNSeparator, "frcrn-pretrain", ROOT / "config" / "frcrn.yaml"),
}


def resolve_url(key: str, url_config: Path = URL_CONFIG) -> str:
    with open(url_config, encoding="utf-8") as f:
        url = yaml.safe_load(f).get(key)
    if not url:
        raise KeyError(f"'{key}' is missing or empty in {url_config}")
    return url


def download_checkpoint(url: str, out_dir: Path = PRETRAINED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / Path(urlparse(url).path).name
    if destination.exists():
        logger.info("Checkpoint already exists: %s", destination)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(partial, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress,
        ):
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
                progress.update(len(chunk))
    partial.rename(destination)
    return destination


def get_spec(name: str) -> PretrainedSpec:
    if name not in PRETRAINED:
        raise KeyError(f"Unknown pretrained model '{name}', expected one of {sorted(PRETRAINED)}")
    return PRETRAINED[name]


def download_pretrained(name: str, url_config: Path = URL_CONFIG, out_dir: Path = PRETRAINED_DIR) -> Path:
    return download_checkpoint(resolve_url(get_spec(name).url_key, url_config), out_dir)


def load_pretrained(
    name: str, url_config: Path = URL_CONFIG, out_dir: Path = PRETRAINED_DIR
) -> tuple[BaseModel, LoadReport]:
    spec = get_spec(name)
    model = spec.model_cls.from_yaml(spec.config_path)
    report = model.load_pretrained(download_pretrained(name, url_config, out_dir))
    logger.info("Loaded %s: missing=%d, unexpected=%d", name, len(report.missing), len(report.unexpected))
    return model, report


def download_frcrn_best(fs: Any, destination: Path = FRCRN_BEST_PATH) -> Path:
    """Fetch the trained FRCRN checkpoint (FRCRN/best.pt) from Google Drive (skips if present)."""
    if destination.is_file():
        logger.info("Checkpoint already exists: %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fs.get_file(f"{DRIVE_FRCRN_DIR}/{BEST_FILENAME}", str(destination))
    return destination


def load_teacher_models(fs: Any, device: torch.device | str = "cpu") -> tuple[FRCRNSeparator, BEATClassifier]:
    """Return frozen (FRCRN from Drive, pretrained BEATs from HuggingFace) on device."""
    separator = FRCRNSeparator.from_yaml(get_spec("frcrn").config_path)
    report = separator.load_pretrained(download_frcrn_best(fs))
    logger.info(
        "Loaded frcrn: missing=%d, unexpected=%d, target=%s",
        len(report.missing),
        len(report.unexpected),
        separator.config.target,
    )

    beats, _ = load_pretrained("beats")
    return separator.to(device).freeze(), beats.to(device).freeze()
