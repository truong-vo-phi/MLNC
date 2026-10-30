from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from scripts.load_pretrain import download_pretrained
from src.model.frcrn import FRCRNSeparator
from src.utils.logger import setup_logger
from src.utils.paths import RunPaths
from src.utils.seed import set_seed

from .config import CEDTinyPipelineConfig, resolve_path, smoke_config
from .data import build_datasets, build_loaders
from .evaluate import (
    collect_predictions,
    compute_metrics,
    compute_snr_breakdown,
    compute_unseen_metrics,
    write_result_csv,
    write_snr_breakdown_csv,
)
from .frontend import LogMel, SpecAugment
from .model import CEDTiny
from .runtime import resolve_device
from .trainer import CEDTinyTrainer


def resolve_checkpoint_path(preferred: str | Path, fallback: str | Path, model_name: str) -> Path:
    p = resolve_path(preferred)
    if p.is_file():
        return p
    fb = resolve_path(fallback)
    if fb.is_file():
        return fb
    return download_pretrained(model_name)


def load_separator(
    config: CEDTinyPipelineConfig, device: torch.device, logger: logging.Logger
) -> FRCRNSeparator | None:
    if config.data.nhat_cache_dir is not None:
        logger.info("Using cached nhat directory: %s", config.data.nhat_cache_dir)
        return None
    frcrn_cfg_path = resolve_path(config.frcrn_config)
    separator = FRCRNSeparator.from_yaml(frcrn_cfg_path)
    ckpt = resolve_checkpoint_path(config.frcrn.checkpoint, config.frcrn.fallback_checkpoint, "frcrn")
    logger.info("Loading FRCRN checkpoint: %s", ckpt)
    separator.load_pretrained(ckpt)
    separator.to(device)
    if config.frcrn.freeze:
        separator.freeze()
    return separator


def load_classifier(config: CEDTinyPipelineConfig, device: torch.device, logger: logging.Logger) -> CEDTiny:
    logger.info("Initializing CED Tiny architecture (%s, init=%s)", config.ced_tiny.arch, config.ced_tiny.init)
    frontend = LogMel(**config.frontend.to_dict())
    spec_augment = SpecAugment(**config.specaugment.to_dict())
    classifier = CEDTiny(config.ced_tiny, frontend, spec_augment)
    classifier.to(device)
    return classifier


def run_pipeline(
    config: CEDTinyPipelineConfig,
    output_dir: Path | None = None,
    run_id: str | None = None,
    smoke: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if smoke:
        config = smoke_config(config)

    set_seed(config.train.seed)
    dev = device or resolve_device()

    base_out = output_dir or (Path("output/test") if smoke else Path("output"))
    run = RunPaths.from_run_id(run_id, base_out) if run_id else RunPaths.new("ced-tiny", base_out)
    run.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("ced-tiny", run.log_file)

    logger.info("Starting CED-Tiny pipeline | run_id=%s | device=%s | smoke=%s", run.run_id, dev, smoke)

    datasets = build_datasets(config.data)
    loaders = build_loaders(
        datasets,
        batch_size=config.train.batch_size,
        eval_batch_size=config.train.eval_batch_size,
        num_workers=config.data.num_workers,
        balanced_sampling=config.data.balanced_sampling,
        seed=config.train.seed,
    )
    logger.info("Datasets loaded: train=%d, validation=%d", len(datasets["train"]), len(datasets["validation"]))

    separator = load_separator(config, dev, logger)
    classifier = load_classifier(config, dev, logger)

    ckpt_path = run.checkpoint_path("ced_tiny")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = run.checkpoint_dir / "resume" / "ced_tiny.pt"
    resume_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = CEDTinyTrainer(
        model=classifier,
        config=config.train,
        device=dev,
        logger=logger,
        name="ced_tiny_frcrn",
        checkpoint_path=ckpt_path,
        separator=separator,
        target_rms=config.frcrn.target_rms,
        resume_path=resume_path,
    )

    if resume_path.is_file():
        trainer.load_resume(resume_path)

    fit_result = trainer.fit(loaders["train"], loaders["validation"])
    trainer.restore_best()

    logger.info("Evaluating best model on validation split")
    val_preds = collect_predictions(
        model=classifier,
        loader=loaders["validation"],
        device=dev,
        separator=separator,
        target_rms=config.frcrn.target_rms,
        amp_mode=config.train.amp,
        desc="val_eval",
    )
    val_overall = compute_metrics(val_preds.probs, val_preds.labels, classifier.config.num_classes)
    val_snr = compute_snr_breakdown(val_preds, config.data.snr_levels, classifier.config.num_classes)

    unseen_overall, unseen_snr, n_unseen = compute_unseen_metrics(
        val_preds,
        config.data.audioset_eval_segments,
        config.data.audioset_train_segments,
        config.data.snr_levels,
        classifier.config.num_classes,
    )

    result_payload: dict[str, Any] = {
        "run_id": run.run_id,
        "best_epoch": fit_result.best_epoch,
        "best_score": fit_result.best_score,
        "epochs_run": fit_result.epochs_run,
        "val_metrics": val_overall,
        "val_by_snr": val_snr,
        "history": trainer.history,
    }

    if unseen_overall is not None:
        result_payload["unseen_metrics"] = unseen_overall
        result_payload["unseen_by_snr"] = unseen_snr
        result_payload["unseen_count"] = n_unseen
        logger.info(
            "Unseen validation evaluated on %d samples: Acc=%.4f, Macro-F1=%.4f",
            n_unseen,
            unseen_overall["top1_accuracy"],
            unseen_overall["macro_f1"],
        )

    run.result_file.parent.mkdir(parents=True, exist_ok=True)
    run.result_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    csv_rows = [{"split": "validation", **val_overall}]
    if unseen_overall is not None:
        csv_rows.append({"split": "validation_unseen", **unseen_overall})
    write_result_csv(run.result_dir / "result.csv", csv_rows)

    write_snr_breakdown_csv(run.result_dir / "snr_breakdown_val.csv", val_snr)
    if unseen_snr is not None:
        write_snr_breakdown_csv(run.result_dir / "snr_breakdown_unseen.csv", unseen_snr)

    logger.info("Pipeline complete. Results saved to %s", run.result_dir)
    return result_payload
