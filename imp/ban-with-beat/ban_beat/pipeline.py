from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from scripts.load_dataset import build_dataloader
from scripts.load_pretrain import download_pretrained
from src.model.beat import BEATClassifier
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import setup_logger
from src.utils.paths import RunPaths
from src.utils.seed import set_seed

from .config import REPO_ROOT, BANConfig, BANLossConfig, load_config, resolve_path, smoke_config
from .data import build_datasets, build_loaders, check_split_disjoint, coverage_report
from .ensemble import EnsembleReport, evaluate_ensemble
from .evaluate import (
    Predictions,
    collect_predictions,
    load_predictions,
    metrics_by_snr,
    plot_snr_comparison,
    predictions_from_cache,
    save_predictions,
    write_result_csv,
)
from .runtime import resolve_device
from .student import BANStudent
from .teacher import (
    BANTeacher,
    TeacherMetadata,
    cache_teacher_logits,
    ensure_teacher_cache,
    load_teacher,
    teacher_fingerprint,
)
from .teacher_finetune import finetune_teacher
from .trainer import BANTrainer


@dataclass
class GenerationResult:
    name: str
    seed: int
    checkpoint: Path
    metrics: dict[str, float]
    history: list[dict[str, float]]
    predictions: Predictions
    resumed: bool = False
    model: BANStudent | None = None


@dataclass
class PipelineResult:
    run: RunPaths
    config: BANConfig
    teacher: dict[str, float]
    teacher_metadata: TeacherMetadata
    teacher_predictions: Predictions
    generations: list[GenerationResult]
    baseline_generations: list[GenerationResult]
    bane: EnsembleReport | None
    baseline: EnsembleReport | None
    split_report: dict[str, Any]
    snr_levels: tuple[float, ...]

    def table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [{"model": "teacher", **self.teacher}]
        rows.extend(
            {"model": generation.name, "seed": generation.seed, **generation.metrics} for generation in self.generations
        )
        if self.bane is not None:
            rows.append(
                {
                    "model": "bane",
                    **self.bane.metrics,
                    **{f"div_{key}": value for key, value in self.bane.diversity.items()},
                }
            )
        rows.extend(
            {"model": generation.name, "seed": generation.seed, **generation.metrics}
            for generation in self.baseline_generations
        )
        if self.baseline is not None:
            rows.append(
                {
                    "model": "baseline_ens",
                    **self.baseline.metrics,
                    **{f"div_{key}": value for key, value in self.baseline.diversity.items()},
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run.run_id,
            "teacher": self.teacher,
            "teacher_metadata": self.teacher_metadata.to_dict(),
            "generations": [
                {
                    "name": generation.name,
                    "seed": generation.seed,
                    "checkpoint": str(generation.checkpoint),
                    "metrics": generation.metrics,
                    "resumed": generation.resumed,
                }
                for generation in self.generations
            ],
            "baseline_generations": [
                {
                    "name": generation.name,
                    "seed": generation.seed,
                    "checkpoint": str(generation.checkpoint),
                    "metrics": generation.metrics,
                    "resumed": generation.resumed,
                }
                for generation in self.baseline_generations
            ],
            "bane": None if self.bane is None else {"metrics": self.bane.metrics, "diversity": self.bane.diversity},
            "baseline": None
            if self.baseline is None
            else {"metrics": self.baseline.metrics, "diversity": self.baseline.diversity},
            "split_report": self.split_report,
        }


def _resolve_teacher_checkpoint(config: BANConfig) -> Path:
    if config.teacher.checkpoint is not None:
        path = resolve_path(config.teacher.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"teacher checkpoint not found: {path}")
        return path
    candidates = (
        REPO_ROOT / "pretrained" / "beat-ft" / "best.pt",
        REPO_ROOT / "pretrained" / "beat-ft" / "beats_teacher_fold0_weights.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no teacher checkpoint found; set teacher.checkpoint in the config (tried {candidates})")


def build_student(
    config: BANConfig,
    seed: int,
    separator: nn.Module | None,
    input_rms: float | None,
) -> BANStudent:
    set_seed(seed)
    classifier = BEATClassifier.from_yaml(config.model_config)
    if config.generations.student_init == "pretrained":
        classifier.load_pretrained(download_pretrained("beats"))
    return BANStudent(classifier, separator, input_rms)


def _make_trainer(
    model: BEATClassifier,
    config: BANConfig,
    loss_config: BANLossConfig,
    teacher: BANTeacher | None,
    teacher_logits: Mapping[str, Tensor] | None,
    device: torch.device,
    logger: logging.Logger,
    name: str,
    run: RunPaths,
) -> BANTrainer:
    resume_path = run.checkpoint_dir / "resume" / f"{name}.pt"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    return BANTrainer(
        model=model,
        config=config.train,
        loss_config=loss_config,
        snr_levels=config.data.snr_levels,
        teacher=teacher,
        teacher_logits=teacher_logits,
        device=device,
        logger=logger,
        name=name,
        checkpoint_path=run.checkpoint_path(name),
        resume_path=resume_path,
    )


def _load_generation(config: BANConfig, name: str, seed: int, run: RunPaths) -> GenerationResult | None:
    summary_path = run.result_dir / f"{name}.json"
    predictions_path = run.result_dir / f"{name}_predictions.npz"
    checkpoint = run.checkpoint_path(name)
    if not (summary_path.is_file() and predictions_path.is_file() and checkpoint.is_file()):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return GenerationResult(
        name=name,
        seed=int(summary.get("seed", seed)),
        checkpoint=checkpoint,
        metrics=summary["metrics"],
        history=summary.get("history", []),
        predictions=load_predictions(predictions_path),
        resumed=True,
    )


def _run_generation(
    config: BANConfig,
    name: str,
    seed: int,
    loss_config: BANLossConfig,
    teacher: BANTeacher | None,
    teacher_logits: Mapping[str, Tensor] | None,
    loaders: dict[str, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run: RunPaths,
) -> GenerationResult:
    cached = _load_generation(config, name, seed, run)
    if cached is not None:
        logger.info("[%s] reused %s", name, run.checkpoint_path(name))
        return cached

    model = build_student(config, seed, teacher.separator, teacher.input_rms)
    trainer = _make_trainer(model, config, loss_config, teacher, teacher_logits, device, logger, name, run)
    if trainer.resume_path is not None and trainer.resume_path.is_file():
        trainer.load_resume(trainer.resume_path)
    fit = trainer.fit(loaders["train"], loaders["validation"])
    trainer._in_fit = False

    predictions = collect_predictions(
        lambda waveform: model(waveform).logits,
        loaders["validation"],
        device,
        config.train.amp,
        desc=f"{name} - validation",
    )
    metrics = metrics_by_snr(predictions, config.data.snr_levels)
    summary = {
        "name": name,
        "seed": seed,
        "checkpoint": str(run.checkpoint_path(name)),
        "best_epoch": fit.best_epoch,
        "best_score": fit.best_score,
        "metrics": metrics,
        "history": trainer.history,
    }
    (run.result_dir / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_predictions(run.result_dir / f"{name}_predictions.npz", predictions)
    logger.info(
        "[%s] seed %d | best epoch %d | top1 %.4f | macro_f1 %.4f | macro_f1_snr %.4f",
        name,
        seed,
        fit.best_epoch,
        metrics["top1_accuracy"],
        metrics["macro_f1"],
        metrics["macro_f1_snr"],
    )
    return GenerationResult(
        name=name,
        seed=seed,
        checkpoint=run.checkpoint_path(name),
        metrics=metrics,
        history=trainer.history,
        predictions=predictions,
        model=model,
    )


def _chain_student(config: BANConfig, generation: GenerationResult, teacher: BANTeacher) -> BANStudent:
    if generation.model is not None:
        return generation.model
    student = build_student(config, generation.seed, teacher.separator, teacher.input_rms)
    load_checkpoint(generation.checkpoint, student)
    return student


def _train_logits(
    teacher: BANTeacher, loader: DataLoader, device: torch.device, amp: str, desc: str
) -> dict[str, Tensor]:
    sample_ids, logits = cache_teacher_logits(teacher, loader, device, amp, desc=desc)
    return {sample_id: torch.from_numpy(row) for sample_id, row in zip(sample_ids, logits, strict=True)}


def _plot_comparison(result: PipelineResult) -> list[Path]:
    series: dict[str, Predictions] = {"teacher": result.teacher_predictions}
    series.update({generation.name: generation.predictions for generation in result.generations})
    if result.bane is not None:
        series["bane"] = result.bane.predictions
    if result.baseline is not None:
        series["baseline_ens"] = result.baseline.predictions
    paths = []
    for metric in ("top1_accuracy", "macro_f1"):
        paths.append(
            plot_snr_comparison(
                series,
                result.snr_levels,
                result.run.figure_dir / f"snr_{metric}.png",
                metric=metric,
                title=f"{metric} by SNR | {result.run.run_id}",
            )
        )
    return paths


def run_pipeline(
    config_path: str | Path | BANConfig,
    output_dir: str | Path = REPO_ROOT / "output",
    device: str | None = None,
    smoke: bool = False,
    run_id: str | None = None,
    num_workers: int | None = None,
    logger: logging.Logger | None = None,
) -> PipelineResult:
    config = config_path if isinstance(config_path, BANConfig) else load_config(config_path)
    if smoke:
        config = smoke_config(config)
    if num_workers is not None:
        config = replace(config, data=replace(config.data, num_workers=num_workers))

    if config.data.nhat_cache_dir is not None and not config.teacher.finetune:
        raise ValueError(
            "data.nhat_cache_dir requires teacher.finetune = true (the cache uses teacher.frcrn_checkpoint)"
        )
    torch_device = resolve_device(device)
    run = (RunPaths.from_run_id(run_id, output_dir) if run_id else RunPaths.new("ban-beat", output_dir)).create()
    log = logger or setup_logger("ban-beat", run.log_file)
    set_seed(config.train.seed)
    log.info("run %s | device %s | smoke %s", run.run_id, torch_device, smoke)
    log.info("config: %s", json.dumps(config.to_dict(), ensure_ascii=False))

    datasets = build_datasets(config.data)
    loaders = build_loaders(
        datasets,
        batch_size=config.train.batch_size,
        eval_batch_size=config.train.eval_batch_size,
        num_workers=config.data.num_workers,
        balanced_sampling=config.data.balanced_sampling,
        seed=config.train.seed,
    )
    overlaps = check_split_disjoint(datasets)
    coverage = coverage_report(datasets, config.data.snr_levels)
    log.info("split overlap: %s", overlaps)
    log.info("class coverage: %s", coverage["class_counts"])
    log.info("snr coverage: %s", coverage["snr_counts"])

    frcrn_config = resolve_path(config.teacher.frcrn_config)
    if config.teacher.finetune:
        teacher, metadata, teacher_path = finetune_teacher(config, loaders, torch_device, log, run)
    else:
        teacher_path = _resolve_teacher_checkpoint(config)
        teacher, metadata = load_teacher(
            teacher_path,
            config.model_config,
            frcrn_config,
            torch_device,
            config.teacher.input_rms,
        )
    log.info("teacher: %s", metadata.to_dict())
    fingerprint = teacher_fingerprint(
        teacher_path,
        [config.model_config, frcrn_config, f"rms={metadata.input_rms}"],
    )
    cache_loaders = {
        "train": build_dataloader(datasets["train"], config.train.eval_batch_size, False, config.data.num_workers),
        "validation": loaders["validation"],
    }
    caches = ensure_teacher_cache(
        teacher,
        cache_loaders,
        datasets,
        resolve_path(config.teacher.cache_dir),
        fingerprint,
        torch_device,
        config.train.amp,
        persist=config.teacher.cache_logits,
    )
    teacher_predictions = predictions_from_cache(datasets["validation"], caches["validation"])
    teacher_metrics = metrics_by_snr(teacher_predictions, config.data.snr_levels)
    log.info(
        "teacher validation | top1 %.4f | macro_f1 %.4f | macro_f1_snr %.4f",
        teacher_metrics["top1_accuracy"],
        teacher_metrics["macro_f1"],
        teacher_metrics["macro_f1_snr"],
    )

    generations: list[GenerationResult] = []
    current_teacher, current_logits = teacher, caches["train"]
    for index in range(config.generations.count):
        seed = int(config.generations.seeds[index])
        generation = _run_generation(
            config,
            f"g{index + 1}",
            seed,
            config.loss,
            current_teacher,
            current_logits,
            loaders,
            torch_device,
            log,
            run,
        )
        generations.append(generation)
        if index + 1 < config.generations.count:
            current_teacher = _chain_student(config, generation, teacher).as_teacher()
            current_logits = _train_logits(
                current_teacher,
                cache_loaders["train"],
                torch_device,
                config.train.amp,
                desc=f"teacher cache g{index + 1}",
            )

    baseline_generations: list[GenerationResult] = []
    baseline: EnsembleReport | None = None
    if config.baseline.enabled:
        baseline_seeds = config.baseline.seeds or config.generations.seeds[: config.generations.count]
        baseline_loss = replace(config.loss, kd_weight=0.0, attention_weight=0.0)
        for index, seed in enumerate(baseline_seeds):
            baseline_generations.append(
                _run_generation(
                    config,
                    f"base{index + 1}",
                    int(seed),
                    baseline_loss,
                    teacher,
                    None,
                    loaders,
                    torch_device,
                    log,
                    run,
                )
            )
        if baseline_generations:
            baseline = evaluate_ensemble(
                [generation.predictions for generation in baseline_generations],
                config.data.snr_levels,
            )

    bane = evaluate_ensemble([generation.predictions for generation in generations], config.data.snr_levels)
    result = PipelineResult(
        run=run,
        config=config,
        teacher=teacher_metrics,
        teacher_metadata=metadata,
        teacher_predictions=teacher_predictions,
        generations=generations,
        baseline_generations=baseline_generations,
        bane=bane if config.generations.ensemble else None,
        baseline=baseline,
        split_report={"overlap": overlaps, "coverage": coverage},
        snr_levels=config.data.snr_levels,
    )

    write_result_csv(run.result_file, result.table())
    (run.result_dir / "summary.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run.result_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    histories = {generation.name: generation.history for generation in [*generations, *baseline_generations]}
    run.history_path("generations").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    _plot_comparison(result)
    log.info("result: %s", run.result_file)
    return result


def load_run(run_id: str, output_dir: str | Path = REPO_ROOT / "output") -> dict[str, Any]:
    run = RunPaths.from_run_id(run_id, output_dir)
    summary = run.result_dir / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"summary not found: {summary}")
    return json.loads(summary.read_text(encoding="utf-8"))
