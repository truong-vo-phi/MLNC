from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune import METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES  # noqa: E402
from scripts.load_dataset import AudioSet40Dataset, Mixture36Dataset, build_dataloader, subsample  # noqa: E402
from scripts.load_pretrain import download_checkpoint, resolve_url  # noqa: E402
from scripts.run_baselines import (  # noqa: E402
    available_frames,
    best_epoch_scores,
    build_loaders,
    count_millions,
    write_result_csv,
)
from scripts.teacher_cache import frcrn_sisdr_by_snr, median_rms  # noqa: E402
from src.model.baseline import BaselineConfig, BaselineModel  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.model.reviewkd import ReviewKDConfig, ReviewModule, ReviewTeacher  # noqa: E402
from src.train import ReviewKDTrainConfig, ReviewKDTrainer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.paths import RunPaths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

REVIEWKD_CONFIG = ROOT / "config" / "reviewkd.yaml"
FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"
BEATS_CONFIG = ROOT / "config" / "beat.yaml"
FBANK_FRAME_MS = 25
FBANK_SHIFT_MS = 10
NOISE_PROBE_CLIPS = 128
RESULT_COLUMNS = (
    "model",
    "review_weight",
    "params_m",
    "best_epoch",
    "epochs_run",
    "train_seconds",
    "macro_acc",
    *METRIC_NAMES,
)


def read_reviewkd_config(path: str | Path = REVIEWKD_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def teacher_grid(duration: float, frontend: dict[str, Any], patch: int = 16, mel_bins: int = 128) -> tuple[int, int]:
    samples = int(duration * frontend["sample_rate"])
    frame = FBANK_FRAME_MS * frontend["sample_rate"] // 1000
    shift = FBANK_SHIFT_MS * frontend["sample_rate"] // 1000
    frames = 1 + (samples - frame) // shift
    return mel_bins // patch, frames // patch


def student_grid(raw: dict[str, Any], stages: int = 4) -> tuple[int, int]:
    scale = 2**stages
    return raw["model"]["freq_bins"] // scale, raw["model"]["num_frames"] // scale


def check_grids(raw: dict[str, Any]) -> tuple[int, int]:
    frames = available_frames(raw["dataset"]["duration"], raw["frontend"])
    if raw["model"]["num_frames"] != frames:
        raise ValueError(
            f"student num_frames {raw['model']['num_frames']} must equal the {frames} frames of the clip so the "
            f"stage-4 grid is not cropped away from the teacher patches"
        )
    teacher = teacher_grid(raw["dataset"]["duration"], raw["frontend"])
    student = student_grid(raw, len(raw["model"]["channels"]))
    if student != teacher:
        raise ValueError(f"student stage grid {student} (freq, time) differs from teacher grid {teacher}")
    return teacher


def build_student(raw: dict[str, Any], load_pretrained: bool = True) -> BaselineModel:
    """load_pretrained=False: không tải/nạp PANNs (khi trọng số sẽ được ghi đè bởi resume.pt)."""
    student = BaselineModel(BaselineConfig.from_dict(raw["model"]), raw["frontend"])
    if student.config.pretrained and load_pretrained:
        student.backbone.load_panns(download_checkpoint(resolve_url("panns-cnn10")))
    return student


def build_review(raw: dict[str, Any], review: ReviewKDConfig, grid: tuple[int, int]) -> ReviewModule:
    return ReviewModule(
        in_channels=raw["model"]["channels"],
        out_channels=review.teacher_dim,
        out_size=grid,
        mid_channels=review.mid_channels,
        resize=review.resize,
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_teacher_models(raw: dict[str, Any], device: torch.device) -> tuple[FRCRNSeparator, BEATClassifier]:
    paths = raw["teacher"]
    separator = FRCRNSeparator.from_yaml(FRCRN_CONFIG)
    separator.load_pretrained(resolve_path(paths["frcrn_checkpoint"]))
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    beats.load_pretrained(resolve_path(paths["beats_checkpoint"]))
    if paths.get("beats_finetuned"):
        load_checkpoint(resolve_path(paths["beats_finetuned"]), beats)
    return separator.to(device).freeze(), beats.to(device).freeze()


def resolve_noise_source(
    separator: FRCRNSeparator,
    configured: bool | None,
    duration: float,
    device: torch.device,
    logger: logging.Logger,
    num_workers: int,
) -> bool:
    if configured is not None:
        return configured
    dataset = Mixture36Dataset(split="validation", duration=duration, load_noise=True)
    loader = build_dataloader(subsample(dataset, NOISE_PROBE_CLIPS), 16, False, num_workers)
    as_output = frcrn_sisdr_by_snr(separator, loader, device, speech_as_noise=False)
    as_residual = frcrn_sisdr_by_snr(separator, loader, device, speech_as_noise=True)
    from_residual = bool(np.mean(list(as_residual.values())) > np.mean(list(as_output.values())))
    logger.info("T1 | SI-SDR n_hat=output %s | n_hat=x-output %s", as_output, as_residual)
    return from_residual


def resolve_target_rms(raw: dict[str, Any], configured: float, max_clips: int | None) -> float:
    if configured > 0:
        return configured
    dataset = AudioSet40Dataset(split="train", duration=raw["dataset"]["duration"])
    return median_rms(dataset, max_clips=max_clips or raw["teacher"]["rms_clips"])


def build_teacher(
    raw: dict[str, Any],
    review: ReviewKDConfig,
    device: torch.device,
    logger: logging.Logger,
    *,
    max_clips: int | None = None,
    num_workers: int = 0,
) -> ReviewTeacher:
    separator, beats = load_teacher_models(raw, device)
    from_residual = resolve_noise_source(
        separator, review.noise_from_residual, raw["dataset"]["duration"], device, logger, num_workers
    )
    target_rms = resolve_target_rms(raw, review.target_rms, max_clips)
    logger.info("teacher | noise_from_residual=%s | target_rms=%.5f", from_residual, target_rms)
    resolved = replace(review, noise_from_residual=from_residual, target_rms=target_rms)
    return ReviewTeacher(separator, beats, resolved, target_rms).to(device)


def train_review(
    raw: dict[str, Any],
    run: RunPaths,
    device: torch.device,
    logger: logging.Logger,
    *,
    review_weight: float | None = None,
    smoke: bool = False,
    max_samples: int | None = None,
    num_workers: int = 4,
    name: str | None = None,
) -> dict[str, Any]:
    grid = check_grids(raw)
    train_config = ReviewKDTrainConfig.from_dict(raw["train"])
    if review_weight is not None:
        train_config = replace(train_config, review_weight=review_weight)
    if smoke:
        train_config = replace(
            train_config,
            epochs=1,
            patience=1,
            batch_size=SMOKE_BATCH,
            eval_batch_size=SMOKE_BATCH,
            review_warmup_epochs=0,
        )
        max_samples, num_workers = SMOKE_SAMPLES, 0
    name = name or ("cnn10-reviewkd" if train_config.review_weight > 0 else "cnn10-ce")

    history_path = run.history_path(name)
    if history_path.is_file():
        logger.info("%s: finished earlier, reading %s", name, history_path)
        return json.loads(history_path.read_text(encoding="utf-8"))["row"]

    set_seed(int(raw["train"].get("seed", 42)))
    loaders = build_loaders(
        raw["dataset"]["name"],
        raw["dataset"]["duration"],
        train_config.batch_size,
        train_config.eval_batch_size,
        max_samples=max_samples,
        num_workers=num_workers,
    )
    review_config = ReviewKDConfig.from_dict(raw["review"])
    student = build_student(raw)
    review_module = build_review(raw, review_config, grid)
    teacher = None
    if train_config.review_weight > 0:
        teacher = build_teacher(raw, review_config, device, logger, max_clips=max_samples, num_workers=num_workers)

    checkpoint_path = run.checkpoint_path(name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    params_m = count_millions(student)
    logger.info(
        "%s | student %.2fM params | review module %.2fM | teacher %s | lambda %.3f | grid %s",
        name,
        params_m,
        count_millions(review_module),
        "yes" if teacher is not None else "no",
        train_config.review_weight,
        grid,
    )

    trainer = ReviewKDTrainer(student, review_module, teacher, train_config, device, logger, name, checkpoint_path)
    set_seed(int(raw["train"].get("seed", 42)))
    started = time.time()
    fit = trainer.fit(loaders["train"], loaders["validation"])
    scores = best_epoch_scores(fit.history, fit.best_epoch)
    row = {
        "model": name,
        "review_weight": train_config.review_weight,
        "params_m": round(params_m, 2),
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        "train_seconds": round(time.time() - started, 1),
        "macro_acc": scores["macro_acc"],
        **{metric: scores[metric] for metric in METRIC_NAMES},
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"row": row, "history": fit.history}, indent=2), encoding="utf-8")
    logger.info("%s: validation | %s", name, " ".join(f"{k} {row[k]:.4f}" for k in ("macro_acc", *METRIC_NAMES)))
    return row


def run_reviewkd(
    run: RunPaths,
    device: torch.device,
    logger: logging.Logger,
    *,
    config_path: str | Path = REVIEWKD_CONFIG,
    include_baseline: bool = True,
    smoke: bool = False,
    max_samples: int | None = None,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    raw = read_reviewkd_config(config_path)
    run.create()
    rows = []
    if include_baseline:
        rows.append(
            train_review(
                raw,
                run,
                device,
                logger,
                review_weight=0.0,
                smoke=smoke,
                max_samples=max_samples,
                num_workers=num_workers,
            )
        )
    rows.append(train_review(raw, run, device, logger, smoke=smoke, max_samples=max_samples, num_workers=num_workers))
    write_result_csv(rows, run.result_file, RESULT_COLUMNS)
    logger.info("results written to %s", run.result_file)
    return rows
