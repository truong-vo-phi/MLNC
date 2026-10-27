from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import DATASETS, find_root  # noqa: E402
from scripts.load_dataset import (  # noqa: E402
    ALL_SNR,
    AUDIOSET_KEY,
    MIXTURE_KEY,
    SAMPLE_RATE,
    VLSP_CONFIG,
    VLSP_KEY,
    AudioSet40Dataset,
    Mixture36Dataset,
    VLSPDataset,
    resolve_dataset_root,
)
from scripts.load_pretrain import get_spec  # noqa: E402
from src.model.kdmoe import KDMoEModel  # noqa: E402

SPLITS = ("train", "validation", "test")
STUDENT_CONFIGS = {"S": "kdmoe_s.yaml", "M": "kdmoe_m.yaml", "L": "kdmoe_l.yaml"}
CLIP_SECONDS = 4.0
TREE_DEPTH = 2


def md_table(rows: Iterable[tuple[Any, Any]], header: tuple[str, str] = ("Tham số", "Giá trị")) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines += [f"| {cell(key)} | {cell(value)} |" for key, value in rows]
    return "\n".join(lines)


def md_grid(header: list[str], rows: Iterable[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows]
    return "\n".join(lines)


def millions(module: nn.Module, trainable_only: bool = False) -> float:
    count = sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
    return count / 1e6


def module_tree(model: nn.Module, depth: int = TREE_DEPTH) -> str:
    total = sum(p.numel() for p in model.parameters()) or 1
    rows = []
    for name, module in model.named_modules():
        if not name or name.count(".") >= depth:
            continue
        count = sum(p.numel() for p in module.parameters())
        rows.append([f"`{name}`", type(module).__name__, f"{count / 1e6:.3f}", f"{100 * count / total:.1f}%"])
    return md_grid(["Module", "Lớp", "Tham số (M)", "Tỉ lệ"], rows)


def full_repr(model: nn.Module) -> str:
    return f"<details><summary>repr đầy đủ</summary>\n\n```text\n{model}\n```\n\n</details>"


def yaml_sections(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    blocks = []
    for section, values in raw.items():
        rows = values.items() if isinstance(values, Mapping) else [(section, values)]
        blocks.append(f"**{section}**\n\n{md_table(rows)}")
    return "\n\n".join(blocks)


def describe_model(model: nn.Module, yaml_path: Path | None = None) -> str:
    parameters = [
        ("lớp", type(model).__name__),
        ("tổng tham số (M)", f"{millions(model):.2f}"),
        ("tham số huấn luyện (M)", f"{millions(model, trainable_only=True):.2f}"),
        ("dung lượng fp32 (MB)", f"{sum(p.numel() for p in model.parameters()) * 4 / 2**20:.0f}"),
    ]
    parts = [md_table(parameters)]
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        parts.append(f"**config model (dataclass)**\n\n{md_table(config.to_dict().items())}")
    if yaml_path is not None:
        parts.append(f"**{yaml_path.name}** (toàn bộ các section)\n\n{yaml_sections(yaml_path)}")
    parts.append(f"**Cấu trúc (độ sâu {TREE_DEPTH})**\n\n{module_tree(model)}")
    parts.append(full_repr(model))
    return "\n\n".join(parts)


@torch.no_grad()
def describe_shapes(separator: nn.Module, beats: nn.Module, seconds: float = CLIP_SECONDS) -> str:
    device = next(separator.parameters()).device
    waveform = torch.zeros(1, int(seconds * SAMPLE_RATE), device=device)
    noise = separator(waveform)
    fbank = beats.frontend(noise)
    features = beats.backbone(fbank)
    logits = beats.head(features)
    rows = [
        ("waveform trộn (x)", tuple(waveform.shape)),
        ("FRCRN: ước lượng " + getattr(separator.config, "target", ""), tuple(noise.shape)),
        ("BEATs Fbank", tuple(fbank.shape)),
        ("BEATs patch features", tuple(features.shape)),
        ("logits", tuple(logits.shape)),
    ]
    flow = "x -> FRCRN -> Fbank -> BEATs backbone -> mean-pool -> head -> logits"
    return f"`{flow}` (clip {seconds:g} s)\n\n{md_table(rows, ('Tensor', 'Shape'))}"


def describe_students(config_dir: Path = ROOT / "config") -> str:
    models = {name: KDMoEModel.from_yaml(config_dir / file) for name, file in STUDENT_CONFIGS.items()}
    keys = list(next(iter(models.values())).config.to_dict())
    rows = [[key, *(model.config.to_dict()[key] for model in models.values())] for key in keys]
    rows.append(["tham số (M)", *(f"{millions(model):.2f}" for model in models.values())])
    parts = [md_grid(["Tham số", *models], rows)]
    for name, file in STUDENT_CONFIGS.items():
        parts.append(f"**{name}: cấu trúc (độ sâu {TREE_DEPTH})**\n\n{module_tree(models[name])}")
        parts.append(f"**{file}: frontend**\n\n{yaml_sections_for(config_dir / file, 'frontend')}")
    return "\n\n".join(parts)


def yaml_sections_for(path: Path, section: str) -> str:
    with open(path, encoding="utf-8") as f:
        return md_table(yaml.safe_load(f)[section].items())


def class_summary(names: list[str]) -> str:
    return f"{len(names)} lớp: " + ", ".join(names)


def resolve_root(key: str, directory: Path | None) -> Path:
    if directory is None:
        return resolve_dataset_root(key)
    marker = DATASETS[key].marker
    return find_root(Path(directory), marker) if marker else Path(directory)


def describe_dataset(key: str, directory: Path | None = None) -> str:
    try:
        root = resolve_root(key, directory)
    except (KeyError, FileNotFoundError) as error:
        return f"Không tìm thấy dataset `{key}`: {error}"

    if key == MIXTURE_KEY:
        datasets = {split: Mixture36Dataset(root=root, split=split) for split in SPLITS}
        first = datasets["train"]
        snrs = Counter(float(row["target_snr_db"]) for row in first.rows)
        rows = [
            ("root", root),
            ("lớp", class_summary(first.class_names)),
            ("mô tả", "mixture = speech + noise (36 lớp nhiễu), nhãn đơn lớp"),
            ("SNR (dB) trong train", dict(sorted(snrs.items()))),
            ("mốc SNR chuẩn", ALL_SNR),
            ("sample rate", first.sample_rate),
            ("độ dài clip (mẫu)", f"{first.num_samples} ({first.num_samples / first.sample_rate:g} s)"),
        ]
    elif key == AUDIOSET_KEY:
        datasets = {split: AudioSet40Dataset(root=root, split=split) for split in SPLITS}
        first = datasets["train"]
        duration = first.num_samples / first.sample_rate
        rows = [
            ("root", root),
            ("lớp", class_summary(first.class_names)),
            ("mô tả", "nhiễu thuần AudioSet (40 lớp), nhãn multi-hot"),
            ("sample rate", first.sample_rate),
            ("độ dài clip (mẫu)", f"{first.num_samples} ({duration:g} s)"),
        ]
    else:
        datasets = {split: VLSPDataset(root=root, split=split) for split in SPLITS}
        with open(VLSP_CONFIG, encoding="utf-8") as f:
            ratios = yaml.safe_load(f)["split"]
        rows = [
            ("root", root),
            ("mô tả", "VLSP-100h: chỉ có audio tiếng nói, split gán cố định theo hash đường dẫn"),
            ("tỉ lệ split", ratios),
            ("sample rate", datasets["train"].sample_rate),
        ]

    sizes = [(split, f"{len(dataset):,}") for split, dataset in datasets.items()]
    return f"{md_table(rows)}\n\n**Số mẫu theo split**\n\n{md_table(sizes, ('Split', 'Số mẫu'))}"


def describe_pipeline(
    separator: nn.Module,
    beat_classifier: nn.Module,
    dataset_dirs: Mapping[str, Path | None] | None = None,
) -> dict[str, str]:
    """Return {section title: markdown} describing teacher models, student configs and datasets."""
    dataset_dirs = dataset_dirs or {}
    report = {
        "Luồng dữ liệu (shape)": describe_shapes(separator, beat_classifier),
        "FRCRN (separator)": describe_model(separator, get_spec("frcrn").config_path),
        "BEATs (teacher)": describe_model(beat_classifier, get_spec("beats").config_path),
        "KD-MoE student (S / M / L)": describe_students(),
    }
    for key in (MIXTURE_KEY, AUDIOSET_KEY, VLSP_KEY):
        report[f"Dataset: {key}"] = describe_dataset(key, dataset_dirs.get(key))
    return report
