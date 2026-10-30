from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from scripts.download_dataset import ensure_dataset
from scripts.dynamic_mix_36l import mixture_class_names
from scripts.finetune import METRIC_NAMES, set_trainable
from scripts.load_dataset import MIXTURE_KEY, Mixture36Dataset, build_dataloader, subsample
from scripts.load_pretrain import download_pretrained
from scripts.run_baselines import write_result_csv
from src.model.beat import BEATClassifier
from src.utils.logger import setup_logger
from src.utils.paths import RunPaths
from src.utils.plots import BLUE, GRID, ORANGE, plot_confusion_matrix
from src.utils.seed import set_seed

from .config import (
    PACKAGE_ROOT,
    REPO_ROOT,
    EMABeatConfig,
    load_config,
    smoke_config,
)
from .data import AudioSet36Dataset, CacheNhatDataset, PairedStudentTeacherDataset
from .evaluate import compare_raw_and_ema, format_comparison_table
from .runtime import resolve_device
from .trainer import EMABeatTrainer

PIPELINE_NAME = "ema-beat"


class SyntheticAudioDataset(Dataset):
    """Synthetic dataset for smoke testing when real audio files are not available."""

    def __init__(self, num_samples: int = 64, num_classes: int = 36, duration: float = 4.0):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sample_rate = 16000
        self.length = int(duration * self.sample_rate)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(idx)
        waveform = torch.randn(self.length, generator=generator) * 0.05
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        class_idx = idx % self.num_classes
        target[class_idx] = 1.0
        return {
            "waveform": waveform,
            "target": target,
            "sample_id": f"synthetic_{idx:05d}",
            "snr_db": torch.tensor(10.0, dtype=torch.float32),
        }


def build_loaders(
    config: EMABeatConfig,
    dataset_root: Path | None = None,
    smoke: bool = False,
) -> dict[str, DataLoader]:
    """Initialize DataLoaders for Student (cache-nhat), Teacher (AudioSet36) and validation."""
    if smoke:
        train_ds = SyntheticAudioDataset(
            num_samples=config.data.max_train_samples or 32, duration=config.data.duration
        )
        val_ds = SyntheticAudioDataset(
            num_samples=config.data.max_eval_samples or 16, duration=config.data.duration
        )
        return {
            "train": DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True),
            "validation": DataLoader(
                val_ds, batch_size=config.train.eval_batch_size, shuffle=False
            ),
        }

    # 1. Check cache-nhat (Student domain)
    cache_dir = Path(config.data.cache_nhat_dir)
    if not cache_dir.is_dir() and (REPO_ROOT / config.data.cache_nhat_dir).is_dir():
        cache_dir = REPO_ROOT / config.data.cache_nhat_dir

    if (cache_dir / "train.npy").is_file():
        # Teacher domain: clean AudioSet noise clips, paired to each cache row by noise_ytid
        audioset_dir = Path(config.data.audioset_dir)
        if not audioset_dir.is_dir() and (REPO_ROOT / config.data.audioset_dir).is_dir():
            audioset_dir = REPO_ROOT / config.data.audioset_dir
        has_audioset = (audioset_dir / "audio").is_dir()

        def _student_teacher(split: str, max_samples: int | None) -> Dataset:
            if has_audioset:
                return PairedStudentTeacherDataset(
                    cache_dir=cache_dir,
                    audioset_dir=audioset_dir,
                    split=split,
                    duration=config.data.duration,
                    max_samples=max_samples,
                )
            return CacheNhatDataset(cache_dir=cache_dir, split=split, max_samples=max_samples)

        train_ds = _student_teacher("train", config.data.max_train_samples)
        val_ds = _student_teacher("validation", config.data.max_eval_samples)
        loaders = {
            "train": DataLoader(
                train_ds,
                batch_size=config.train.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            ),
            "validation": DataLoader(
                val_ds,
                batch_size=config.train.eval_batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=torch.cuda.is_available(),
            ),
        }

        # 2. AudioSet validation (clean Teacher domain)
        if has_audioset and (audioset_dir / "metadata" / "manifest.csv").is_file():
            teacher_val_ds = AudioSet36Dataset(
                root=audioset_dir,
                split="validation",
                duration=config.data.duration,
                max_samples=config.data.max_eval_samples,
            )
            loaders["audioset_val"] = DataLoader(
                teacher_val_ds,
                batch_size=config.train.eval_batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        return loaders

    # Fallback to Mixture36Dataset if cache-nhat not found
    if dataset_root is None or not dataset_root.exists():
        return build_loaders(config, smoke=True)

    def _loader(split: str, shuffle: bool, max_samples: int | None, sample_ids_file: Path | None = None):
        ds = Mixture36Dataset(
            dataset_root,
            split=split,
            duration=config.data.duration,
            load_noise=False,
            sample_ids_file=sample_ids_file,
        )
        return build_dataloader(
            subsample(ds, max_samples),
            config.train.batch_size if shuffle else config.train.eval_batch_size,
            shuffle,
            config.data.num_workers,
        )

    return {
        "train": _loader("train", True, config.data.max_train_samples),
        "validation": _loader("validation", False, config.data.max_eval_samples),
    }


def plot_dual_curves(history: list[dict[str, Any]], out_path: Path) -> None:
    """Plot training curves comparing loss, macro_F1, and weight distance."""
    if not history:
        return
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    student_f1 = [h.get("val_student_macro_f1", h.get("val_raw_macro_f1", 0.0)) for h in history]
    teacher_f1 = [h.get("val_teacher_macro_f1", h.get("val_ema_macro_f1", 0.0)) for h in history]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    distances = [h.get("weight_distance", 0.0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Ax 1: Training Loss
    axes[0].plot(epochs, train_loss, marker="o", color="tab:blue", label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Ax 2: Macro F1 Comparison (Student vs Teacher EMA)
    axes[1].plot(epochs, student_f1, marker="o", color=BLUE, label="Student (cache-nhat)")
    axes[1].plot(epochs, teacher_f1, marker="s", color=ORANGE, linewidth=2, label="Teacher EMA")
    axes[1].set_title("Validation Macro-F1 (Student vs Teacher)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Ax 3: L2 Weight Distance ||theta_student - theta_teacher||
    axes[2].plot(epochs, distances, marker="^", color="purple", label="||Student - Teacher|| L2")
    axes[2].set_title("Weight Distance (Student vs EMA Teacher)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pipeline(
    config: EMABeatConfig,
    output_dir: Path | None = None,
    run_id: str | None = None,
    smoke: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run full Student-Teacher BEATs training pipeline with EMA."""
    if smoke:
        config = smoke_config(config)

    set_seed(config.train.seed)
    actual_device = device or resolve_device()

    out_base = output_dir or (REPO_ROOT / "output" if not smoke else REPO_ROOT / "output" / "test")
    p_name = PIPELINE_NAME if not smoke else f"{PIPELINE_NAME}-smoke"

    if run_id:
        run = RunPaths.from_run_id(run_id, out_base)
    else:
        run = RunPaths.new(p_name, out_base)
    run.create()

    logger = setup_logger(p_name, run.log_file)
    logger.info("run %s | device %s | smoke=%s", run.run_id, actual_device, smoke)

    # 1. Prepare data
    dataset_root = None
    if not smoke:
        try:
            dataset_root = ensure_dataset(config.data.dataset_key, REPO_ROOT / "datasets")
        except Exception as e:
            logger.warning("Could not ensure dataset automatically: %s.", e)

    loaders = build_loaders(config, dataset_root, smoke=smoke)
    logger.info("DataLoaders initialized. Train batches: %d", len(loaders["train"]))

    # 2. Build Student & Teacher BEATs models (Both load Pretrained BEATs)
    model_cfg_path = REPO_ROOT / config.model_config
    if not model_cfg_path.is_file():
        model_cfg_path = PACKAGE_ROOT / "config" / "beat.yaml"
        if not model_cfg_path.is_file():
            model_cfg_path = REPO_ROOT / "config" / "beat.yaml"

    pretrain_path = None
    if not smoke:
        try:
            pretrain_path = download_pretrained("beats")
        except Exception as e:
            logger.warning("Could not download/locate pretrain weights: %s", e)

    # Student: Load pretrained BEATs
    student = BEATClassifier.from_yaml(model_cfg_path)
    if pretrain_path is not None:
        try:
            report = student.load_pretrained(pretrain_path)
            logger.info("Loaded pretrain BEATs into Student from %s (missing: %d)", pretrain_path, len(report.missing))
        except Exception as e:
            logger.warning("Could not load pretrain weights for Student: %s", e)
    student = set_trainable(student.to(actual_device), True)

    # Teacher: Load pretrained BEATs checkpoint
    teacher = BEATClassifier.from_yaml(model_cfg_path)
    if pretrain_path is not None:
        try:
            report = teacher.load_pretrained(pretrain_path)
            logger.info("Loaded pretrain BEATs into Teacher from %s (missing: %d)", pretrain_path, len(report.missing))
        except Exception as e:
            logger.warning("Could not load pretrain weights for Teacher: %s", e)
    teacher = set_trainable(teacher.to(actual_device), False)

    # 3. Initialize Trainer
    checkpoint_path = run.checkpoint_path("beats_ema")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_path.with_name("resume.pt")

    audioset_loader = loaders.get("audioset_val")

    trainer = EMABeatTrainer(
        model=student,
        teacher_model=teacher,
        config=config.train,
        ema_config=config.ema,
        device=actual_device,
        logger=logger,
        name="beats_ema",
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
        audioset_val_loader=audioset_loader,
    )

    if resume_path.is_file():
        trainer.load_resume(resume_path)

    # 4. Training
    start_time = time.time()
    fit_result = trainer.fit(loaders["train"], loaders["validation"])
    train_seconds = time.time() - start_time
    logger.info("Training completed in %.1f seconds", train_seconds)

    # 5. Final evaluation on all validation splits (Student & Teacher EMA)
    trainer._in_fit = False
    final_scores: dict[str, dict[str, float]] = {}
    for split_name, loader in loaders.items():
        if split_name == "train":
            continue
        student_eval = trainer._evaluate_with(trainer.model.predict_proba, loader, desc=f"Final Student - {split_name}")
        teacher_eval = trainer._evaluate_with(trainer.ema.predict_proba, loader, desc=f"Final Teacher - {split_name}")
        final_scores[split_name] = {
            **{f"student_{k}": v for k, v in student_eval.items()},
            **{f"teacher_{k}": v for k, v in teacher_eval.items()},
            **teacher_eval,  # Default metric reported is Teacher EMA
        }

    # 6. Save results CSV and history JSON
    rows = [{"split": name, **values} for name, values in final_scores.items()]
    cols = tuple(rows[0].keys()) if rows else RESULT_COLUMNS
    write_result_csv(rows, run.result_file, columns=cols)

    history_json = run.result_dir / "history" / "training_history.json"
    history_json.parent.mkdir(parents=True, exist_ok=True)
    history_json.write_text(json.dumps(trainer.history, indent=2), encoding="utf-8")

    # 7. Plot comparison curves
    fig_path = run.figure_dir / "training_curves_student_vs_teacher.png"
    plot_dual_curves(trainer.history, fig_path)
    logger.info("Saved comparison figure to: %s", fig_path)

    # 8. Export standalone checkpoint with Teacher EMA weights
    standalone_path = checkpoint_path.with_name("beats_teacher_ema_standalone.pt")
    standalone_model = trainer.export_inference_model()
    torch.save({"model": standalone_model.state_dict(), "config": standalone_model.config}, standalone_path)
    logger.info("Exported standalone teacher EMA model to: %s", standalone_path)

    return {
        "run_id": run.run_id,
        "best_score": fit_result.best_score,
        "best_epoch": fit_result.best_epoch,
        "epochs_run": fit_result.epochs_run,
        "train_seconds": train_seconds,
        "scores": final_scores,
        "checkpoint_path": checkpoint_path,
        "standalone_path": standalone_path,
        "figure_path": fig_path,
    }
