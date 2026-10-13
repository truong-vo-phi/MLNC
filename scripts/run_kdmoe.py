from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.drive_sync import DriveFileBackup  # noqa: E402
from scripts.finetune import METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES, StageFiles, prepare_stage  # noqa: E402
from scripts.load_dataset import (  # noqa: E402
    AudioSet40Dataset,
    KDMoEMixDataset,
    Mixture36Dataset,
    PairedClipSampler,
    build_dataloader,
    subsample,
)
from scripts.teacher_cache import (  # noqa: E402
    accuracy_by_snr,
    cross_fit_probe,
    frcrn_sisdr_by_snr,
    median_rms,
    save_teacher_cache,
    teacher_embeddings,
)
from src.loss import KDMoELossConfig, load_membership  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.model.kdmoe import KDMoEModel  # noqa: E402
from src.train import KDMoETrainConfig, KDMoETrainer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

SPLITS = ("train", "validation")  # the test split is never used
STUDENT_CONFIG = ROOT / "config" / "kdmoe_s.yaml"
FINETUNE_DIR = ROOT / "pretrained" / "finetune"
TEACHER_CACHE_DIR = FINETUNE_DIR / "teacher_cache"
DRIVE_TEACHER_DIR = "FA26_Capstone/Truong/kd-moe-ft/teacher_cache"
DRIVE_STUDENT_DIR = "FA26_Capstone/Truong/kd-moe-ft-student"
CLIP_SECONDS = 4.0


def restore_from_drive(fs: Any, remote_dir: str, local_dir: Path, names: list[str], logger: logging.Logger) -> None:
    if fs is None:
        return
    for name in names:
        local = local_dir / name
        try:
            if not local.is_file() and fs.exists(f"{remote_dir}/{name}"):
                fs.get_file(f"{remote_dir}/{name}", str(local))
                logger.info("drive: restored %s/%s", remote_dir, name)
        except Exception as error:
            logger.warning("drive: cannot restore %s: %r", name, error)


def load_finetuned(
    separator: FRCRNSeparator,
    beats: BEATClassifier,
    device: torch.device,
    logger: logging.Logger,
    *,
    ft_dir: str | Path = FINETUNE_DIR,
    fs: Any = None,
) -> tuple[FRCRNSeparator, BEATClassifier]:
    """Freeze the already fine-tuned FRCRN and load the fine-tuned BEATs best.pt (from Drive if missing)."""
    beats_files = prepare_stage("beats", ft_dir, logger, fs)
    if not beats_files.best.is_file():
        raise FileNotFoundError(f"fine-tuned BEATs not found: {beats_files.best} - run the finetune cell first")
    load_checkpoint(beats_files.best, beats)
    logger.info("loaded fine-tuned BEATs (%s); FRCRN used as is", beats_files.best)
    return separator.to(device).freeze(), beats.to(device).freeze()


def build_teacher_cache(
    separator: FRCRNSeparator,
    beats: BEATClassifier,
    device: torch.device,
    logger: logging.Logger,
    *,
    cache_dir: str | Path = TEACHER_CACHE_DIR,
    fs: Any = None,
    max_samples: int | None = None,
    num_workers: int = 4,
    batch_size: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    """p_t of the fine-tuned teacher: x -> FRCRN -> RMS-norm -> BEATs embedding -> 5-fold cross-fit probe.

    Writes teacher_{split}.npz and teacher_report.json into cache_dir (a new dir, so the cache of the
    earlier teacher is untouched); skips everything when all files are present.
    """
    if smoke:
        max_samples, num_workers, batch_size = SMOKE_SAMPLES, 0, 16
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = [f"teacher_{split}.npz" for split in SPLITS] + ["teacher_report.json"]
    restore_from_drive(fs, DRIVE_TEACHER_DIR, cache_dir, names, logger)
    report_path = cache_dir / "teacher_report.json"
    if all((cache_dir / name).is_file() for name in names):
        logger.info("teacher cache already built: %s", cache_dir)
        return json.loads(report_path.read_text(encoding="utf-8"))

    def loader_for(split: str, load_noise: bool):
        dataset = Mixture36Dataset(split=split, duration=CLIP_SECONDS, load_noise=load_noise)
        chosen = subsample(dataset, max_samples)
        rows = [dataset.rows[i] for i in chosen.indices] if hasattr(chosen, "indices") else dataset.rows
        return rows, build_dataloader(chosen, batch_size, False, num_workers)

    separator.to(device).eval()
    beats.to(device).eval()

    _, validation_loader = loader_for("validation", load_noise=True)
    as_output = frcrn_sisdr_by_snr(separator, validation_loader, device, speech_as_noise=False)
    as_residual = frcrn_sisdr_by_snr(separator, validation_loader, device, speech_as_noise=True)
    noise_from_residual = bool(np.mean(list(as_residual.values())) > np.mean(list(as_output.values())))
    logger.info("T1 | n_hat = FRCRN output: %s", {k: round(v, 2) for k, v in as_output.items()})
    logger.info("T1 | n_hat = x - output: %s", {k: round(v, 2) for k, v in as_residual.items()})

    noise_train = AudioSet40Dataset(split="train", duration=CLIP_SECONDS)
    target_rms = median_rms(noise_train, max_clips=max_samples)
    logger.info("T3 | noise RMS target r = %.5f", target_rms)

    membership = np.asarray(load_membership())
    embeddings: dict[str, np.ndarray] = {}
    sample_ids: dict[str, list[str]] = {}
    meta: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        rows, loader = loader_for(split, load_noise=False)
        embeddings[split], sample_ids[split] = teacher_embeddings(
            separator, beats, loader, device, target_rms, speech_as_noise=noise_from_residual
        )
        labels = np.array([int(np.argmax(json.loads(row["multi_hot_36"]))) for row in rows])
        meta[split] = {
            "label": labels,
            "group": membership[labels].argmax(axis=1),
            "ytid": np.array([row["noise_ytid"] for row in rows]),
            "snr": np.array([float(row["target_snr_db"]) for row in rows], dtype=np.float32),
        }
        logger.info("T3 | %s: %d clips, embedding %s", split, len(sample_ids[split]), embeddings[split].shape)

    train_meta = meta["train"]
    oof_logits, probe = cross_fit_probe(embeddings["train"], train_meta["label"], train_meta["ytid"])
    for split in SPLITS:
        logits = oof_logits if split == "train" else probe.logits(embeddings[split])
        save_teacher_cache(cache_dir / f"teacher_{split}.npz", sample_ids[split], logits, meta[split]["snr"])

    report = accuracy_by_snr(oof_logits, train_meta["label"], train_meta["group"], train_meta["snr"], membership)
    report["noise_from_residual"] = noise_from_residual
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("T4 | out-of-fold: %s", {k: round(report[k], 3) for k in METRIC_NAMES})

    if fs is not None:
        for name in names:
            DriveFileBackup(fs, DRIVE_TEACHER_DIR, name, logger)(cache_dir / name)
    return report


def student_loaders(
    cache_dir: Path, config: KDMoETrainConfig, *, max_samples: int | None, num_workers: int
) -> dict[str, Any]:
    datasets = {
        split: subsample(KDMoEMixDataset(split=split, teacher_cache=cache_dir / f"teacher_{split}.npz"), max_samples)
        for split in SPLITS
    }
    sampler = PairedClipSampler(datasets["train"], units_per_batch=32, seed=config.seed)
    return {
        "train": build_dataloader(datasets["train"], None, num_workers=num_workers, batch_sampler=sampler),
        "validation": build_dataloader(datasets["validation"], config.eval_batch_size, num_workers=num_workers),
    }


def run_stage(
    model: KDMoEModel,
    stage: int,
    files: StageFiles,
    loaders: dict[str, Any],
    config: KDMoETrainConfig,
    loss_config: KDMoELossConfig,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    if files.result.is_file():
        model.load_pretrained(files.best)
        logger.info("kdmoe stage %d: finished earlier, loaded %s", stage, files.best)
        return json.loads(files.result.read_text(encoding="utf-8"))

    trainer = KDMoETrainer(
        model,
        config,
        loss_config,
        load_membership(),
        device,
        logger,
        f"kdmoe-stage{stage}",
        files.best,
        stage=stage,
        resume_path=files.resume,
        drive_backup=files.drive_backup,
        on_checkpoint=files.on_checkpoint,
    )
    if files.resume.is_file():
        trainer.load_resume(files.resume)
    fit = trainer.fit(loaders["train"], loaders["validation"])
    result: dict[str, Any] = {**fit.to_dict(), "history": trainer.history}

    if stage == 2:
        trainer._in_fit = False  # final evaluation must not touch history / resume
        result["val"] = trainer.evaluate(loaders["validation"], desc="kdmoe - validation")
        logger.info("kdmoe: validation | " + " ".join(f"{name} {result['val'][name]:.4f}" for name in METRIC_NAMES))
    files.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if files.result_backup is not None:
        files.result_backup(files.result)
    return result


def train_student(
    device: torch.device,
    logger: logging.Logger,
    *,
    config_path: str | Path = STUDENT_CONFIG,
    cache_dir: str | Path = TEACHER_CACHE_DIR,
    out_dir: str | Path = FINETUNE_DIR / "student",
    fs: Any = None,
    drive_dir: str = DRIVE_STUDENT_DIR,
    max_samples: int | None = None,
    num_workers: int = 4,
    smoke: bool = False,
) -> dict[str, Any]:
    """KD-MoE student: stage 1 (L_front) then stage 2 (all losses), resumable; validation with 9 metrics.

    smoke=True: 1 epoch per stage on SMOKE_SAMPLES clips per split, no workers (use a separate out_dir).
    """
    config_path = Path(config_path)
    config = KDMoETrainConfig.from_yaml(config_path)
    if smoke:
        config = replace(config, epochs=1, patience=1, stage1_epochs=1, batch_size=SMOKE_BATCH)
        max_samples, num_workers = SMOKE_SAMPLES, 0
    loss_config = KDMoELossConfig.from_yaml(config_path)
    set_seed(config.seed)
    model = KDMoEModel.from_yaml(config_path)
    loaders = student_loaders(Path(cache_dir), config, max_samples=max_samples, num_workers=num_workers)

    stage1_files = prepare_stage("stage1", out_dir, logger, fs, drive_dir)
    stage2_files = prepare_stage("stage2", out_dir, logger, fs, drive_dir)

    stage1 = run_stage(model, 1, stage1_files, loaders, config, loss_config, device, logger)
    if not stage2_files.resume.is_file() and not stage2_files.result.is_file():
        model.load_pretrained(stage1_files.best)
        logger.info("kdmoe: stage 2 initialised from stage 1 best (%s)", stage1_files.best)
    stage2 = run_stage(model, 2, stage2_files, loaders, config, loss_config, device, logger)
    return {"stage1": stage1, "stage2": stage2}
