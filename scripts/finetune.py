from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.drive_sync import DriveCheckpointUploader, DriveFileBackup  # noqa: E402
from scripts.load_dataset import Mixture36Dataset, build_dataloader, subsample  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.train import BEATTrainConfig, BEATTrainer, FRCRNTrainConfig, FRCRNTrainer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402

FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"
BEAT_CONFIG = ROOT / "config" / "beat.yaml"
CLIP_SECONDS = 4.0
SMOKE_SAMPLES = 64
SMOKE_BATCH = 8
DRIVE_FINETUNE_DIR = "FA26_Capstone/Truong/kd-moe-ft"
METRIC_NAMES = (
    "top1_accuracy",
    "top3_accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "micro_f1",
    "mAP",
    "macro_auc",
)


@dataclass
class StageFiles:
    """Local files of one fine-tune stage plus the Drive hooks that mirror them."""

    best: Path
    resume: Path
    result: Path
    on_checkpoint: Callable[[Path, int, float], None] | None = None
    drive_backup: Callable[[str | Path], None] | None = None
    result_backup: Callable[[str | Path], None] | None = None


def prepare_stage(
    name: str,
    local_dir: str | Path,
    logger: logging.Logger,
    fs: Any = None,
    drive_dir: str = DRIVE_FINETUNE_DIR,
    keep_history: bool = False,
) -> StageFiles:
    """Create the stage files; with fs, pull missing ones from Drive (to resume) and mirror new ones."""
    local_dir = Path(local_dir) / name
    local_dir.mkdir(parents=True, exist_ok=True)
    files = StageFiles(local_dir / "best.pt", local_dir / "resume.pt", local_dir / "result.json")
    if fs is None:
        return files

    remote_dir = f"{drive_dir.strip('/')}/{name}"
    for path in (files.best, files.resume, files.result):
        if path.is_file():
            continue
        try:
            if fs.exists(f"{remote_dir}/{path.name}"):
                fs.get_file(f"{remote_dir}/{path.name}", str(path))
                logger.info("drive: restored %s/%s", remote_dir, path.name)
        except Exception as error:
            logger.warning("drive: cannot restore %s: %r", path.name, error)

    files.on_checkpoint = DriveCheckpointUploader(fs, remote_dir, logger, keep_history)
    files.drive_backup = DriveFileBackup(fs, remote_dir, files.resume.name, logger)
    files.result_backup = DriveFileBackup(fs, remote_dir, files.result.name, logger)
    return files


def set_trainable(model: nn.Module, trainable: bool) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    return model


def mixture_loaders(
    batch_size: int,
    eval_batch_size: int,
    *,
    train_seconds: float,
    eval_seconds: float,
    load_noise: bool,
    max_samples: int | None,
    num_workers: int,
) -> dict[str, Any]:
    """Loaders train/validation on Mixture-36l (all 6 SNR levels); the test split is never used."""

    def build(split: str, seconds: float, size: int, shuffle: bool):
        dataset = Mixture36Dataset(split=split, duration=seconds, load_noise=load_noise)
        return build_dataloader(subsample(dataset, max_samples), size, shuffle, num_workers)

    return {
        "train": build("train", train_seconds, batch_size, True),
        "validation": build("validation", eval_seconds, eval_batch_size, False),
    }


def save_result(files: StageFiles, result: dict[str, Any]) -> None:
    files.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if files.result_backup is not None:
        files.result_backup(files.result)


def finetune_frcrn(
    separator: FRCRNSeparator,
    device: torch.device,
    logger: logging.Logger,
    files: StageFiles,
    *,
    config: FRCRNTrainConfig | None = None,
    max_samples: int | None = None,
    num_workers: int = 4,
    smoke: bool = False,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    make_loaders: Callable[[FRCRNTrainConfig, int | None, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fine-tune FRCRN on Mixture-36l noise targets, resuming from files.resume; leaves it frozen on device.

    A finished stage (files.result exists) only reloads files.best.
    smoke=True: 1 epoch on SMOKE_SAMPLES clips per split, no workers (use a separate stage dir).
    make_loaders(config, max_samples, num_workers): train/validation loaders replacing Mixture-36l's own noise
    (e.g. the AudioSet noise reference); it receives the config and limits after the smoke adjustment.
    """
    config = config or FRCRNTrainConfig.from_yaml(FRCRN_CONFIG)
    if smoke:
        config = replace(config, epochs=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
        max_samples, num_workers = SMOKE_SAMPLES, 0
    if files.result.is_file():
        load_checkpoint(files.best, separator.network)
        separator.to(device).freeze()
        logger.info("frcrn: finished earlier, loaded %s", files.best)
        return json.loads(files.result.read_text(encoding="utf-8"))

    if make_loaders is None:
        loaders = mixture_loaders(
            config.batch_size,
            config.eval_batch_size,
            train_seconds=config.segment_seconds,
            eval_seconds=config.eval_segment_seconds,
            load_noise=True,
            max_samples=max_samples,
            num_workers=num_workers,
        )
    else:
        loaders = make_loaders(config, max_samples, num_workers)
    set_trainable(separator, True)
    trainer = FRCRNTrainer(
        separator,
        config,
        device,
        logger,
        "frcrn",
        files.best,
        files.on_checkpoint,
        files.resume,
        files.drive_backup,
        epoch_callback=on_epoch,
    )
    if files.resume.is_file():
        trainer.load_resume(files.resume)
    fit = trainer.fit(loaders["train"], loaders["validation"])
    logger.info(
        "frcrn: best %s %.2f dB @ epoch %d / %d", fit.selection_metric, fit.best_score, fit.best_epoch, fit.epochs_run
    )

    result = fit.to_dict()
    save_result(files, result)
    separator.to(device).freeze()
    return result


def finetune_beats(
    beats: BEATClassifier,
    separator: FRCRNSeparator,
    device: torch.device,
    logger: logging.Logger,
    files: StageFiles,
    *,
    config: BEATTrainConfig | None = None,
    max_samples: int | None = None,
    num_workers: int = 4,
    smoke: bool = False,
) -> dict[str, Any]:
    """Fine-tune BEATs on Mixture-36l (all SNR) behind the frozen FRCRN; resumes from files.resume.

    Result holds the 9 validation metrics of the best epoch, its confusion matrix and the per-epoch history.
    smoke=True: 1 epoch on SMOKE_SAMPLES clips per split, no workers (use a separate stage dir).
    """
    config = config or BEATTrainConfig.from_yaml(BEAT_CONFIG)
    if smoke:
        config = replace(config, epochs=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
        max_samples, num_workers = SMOKE_SAMPLES, 0
    if files.result.is_file():
        load_checkpoint(files.best, beats)
        logger.info("beat: finished earlier, loaded %s", files.best)
        return json.loads(files.result.read_text(encoding="utf-8"))

    loaders = mixture_loaders(
        config.batch_size,
        config.eval_batch_size,
        train_seconds=CLIP_SECONDS,
        eval_seconds=CLIP_SECONDS,
        load_noise=False,
        max_samples=max_samples,
        num_workers=num_workers,
    )
    set_trainable(beats, True)
    trainer = BEATTrainer(
        beats,
        config,
        device,
        logger,
        "beat_mask",
        files.best,
        separator.freeze(),
        files.resume,
        files.drive_backup,
        files.on_checkpoint,
    )
    if files.resume.is_file():
        trainer.load_resume(files.resume)

    fit = trainer.fit(loaders["train"], loaders["validation"])
    trainer._in_fit = False  # final evaluation must not append to the training history
    val = trainer.evaluate(loaders["validation"], desc="beat_mask - validation")
    confusion = trainer.confusion_matrix(loaders["validation"], desc="beat_mask - confusion")
    logger.info("beat: validation | " + " ".join(f"{name} {val[name]:.4f}" for name in METRIC_NAMES))

    result = {**fit.to_dict(), "history": trainer.history, "val": val, "confusion_matrix": confusion.tolist()}
    save_result(files, result)
    return result
