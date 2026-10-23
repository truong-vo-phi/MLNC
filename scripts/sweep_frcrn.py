from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audioset_noise_ref import make_audioset_noise_loaders  # noqa: E402
from scripts.drive_sync import DRIVE_FRCRN_DIR, DRIVE_FRCRN_VLSP_DIR  # noqa: E402
from scripts.finetune import (  # noqa: E402
    CLIP_SECONDS,
    DRIVE_FINETUNE_DIR,
    FRCRN_CONFIG,
    finetune_frcrn,
    prepare_stage,
)
from scripts.load_dataset import Mixture36Dataset, build_dataloader, subsample  # noqa: E402
from src.loss.frcrn import FRCRNLoss, si_snr  # noqa: E402
from src.metrics import SnrBucketMeter  # noqa: E402
from src.model.frcrn import FRCRNConfig, FRCRNSeparator  # noqa: E402
from src.train import FRCRNTrainConfig  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

SWEEP_CONFIG = ROOT / "config" / "frcrn_sweep.yaml"
ORIGINAL_CHECKPOINT = ROOT / "pretrained" / "last_best_checkpoint.pt"
DRIVE_SWEEP_DIR = "FA26_Capstone/Truong/frcrn-fine-tune"
RESERVED_DRIVE_DIRS = (DRIVE_FRCRN_DIR, DRIVE_FRCRN_VLSP_DIR, DRIVE_FINETUNE_DIR)
SELECTION_METRIC = "noise_si_snr_hi"
PROCESS_MARKER = "run_from_spec"
SAMPLE_RATE = 16000
TRAIN_FIELDS = (
    "lr",
    "lr_decay",
    "halve_lr_after",
    "batch_size",
    "segment_seconds",
    "grad_clip",
    "epochs",
    "patience",
)
EXTRA_FIELDS = ("mask_weight", "seed", "target")

logger = logging.getLogger(__name__)


def load_sweep_config(path: str | Path = SWEEP_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_drive_dir(drive_dir: str) -> str:
    cleaned = drive_dir.strip("/")
    if cleaned in {path.strip("/") for path in RESERVED_DRIVE_DIRS}:
        raise ValueError(f"{cleaned!r} is an existing Drive folder, choose a new one such as {DRIVE_SWEEP_DIR!r}")
    return cleaned


def trial_train_config(
    base: FRCRNTrainConfig, settings: Mapping[str, Any], high_snr_levels: Iterable[float] = (15.0, 20.0)
) -> FRCRNTrainConfig:
    unknown = set(settings) - set(TRAIN_FIELDS) - set(EXTRA_FIELDS)
    if unknown:
        raise ValueError(f"unknown sweep settings: {sorted(unknown)}")
    values = {key: settings[key] for key in TRAIN_FIELDS if key in settings}
    return replace(
        base, **values, selection_metric=SELECTION_METRIC, high_snr_levels=tuple(float(x) for x in high_snr_levels)
    )


def build_trial_separator(
    checkpoint: str | Path = ORIGINAL_CHECKPOINT, mask_weight: float = 1.0, target: str | None = None
) -> FRCRNSeparator:
    with open(FRCRN_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    values = {**raw[FRCRNConfig.section], **({"target": target} if target else {})}
    config = FRCRNConfig.from_dict(values)
    criterion = FRCRNLoss(config, **{**raw.get("loss", {}), "mask_weight": mask_weight})
    separator = FRCRNSeparator(config, criterion)
    report = separator.load_pretrained(checkpoint)
    if not report.is_clean:
        raise ValueError(f"{checkpoint}: missing {list(report.missing[:5])}, unexpected {list(report.unexpected[:5])}")
    return separator


def best_epoch_scores(result: Mapping[str, Any]) -> dict[str, float]:
    best_epoch = int(result["best_epoch"])
    rows = [row for row in result.get("history", []) if int(row["epoch"]) == best_epoch]
    if not rows:
        return {}
    return {key.removeprefix("val_"): value for key, value in rows[0].items() if key.startswith("val_")}


def run_trial(
    settings: Mapping[str, Any],
    *,
    device: torch.device,
    trial_dir: str | Path,
    checkpoint: str | Path = ORIGINAL_CHECKPOINT,
    base_config: FRCRNTrainConfig | None = None,
    high_snr_levels: Iterable[float] = (15.0, 20.0),
    max_samples: int | None = None,
    num_workers: int = 4,
    smoke: bool = False,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    stage_name: str = "frcrn",
    fs: Any = None,
    drive_dir: str = DRIVE_SWEEP_DIR,
    make_loaders: Callable[[FRCRNTrainConfig, int | None, int], dict[str, Any]] | None = None,
    keep_history: bool = False,
) -> dict[str, Any]:
    drive_dir = check_drive_dir(drive_dir)
    base = base_config or FRCRNTrainConfig.from_yaml(FRCRN_CONFIG)
    config = trial_train_config(base, settings, high_snr_levels)
    set_seed(int(settings.get("seed", 0)))
    separator = build_trial_separator(checkpoint, float(settings.get("mask_weight", 1.0)), settings.get("target"))
    files = prepare_stage(stage_name, trial_dir, logger, fs, drive_dir, keep_history)
    return finetune_frcrn(
        separator,
        device,
        logger,
        files,
        config=config,
        max_samples=max_samples,
        num_workers=num_workers,
        smoke=smoke,
        on_epoch=on_epoch,
        make_loaders=make_loaders,
    )


def run_tracked_trial(
    settings: Mapping[str, Any],
    *,
    project: str,
    run_name: str,
    run_id: str | None = None,
    **trial_kwargs: Any,
) -> dict[str, Any]:
    import wandb

    run = wandb.init(project=project, name=run_name, id=run_id, resume="allow", config=dict(settings))
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    try:
        result = run_trial(
            settings,
            on_epoch=lambda epoch, scores: run.log({"epoch": epoch, **scores}),
            **trial_kwargs,
        )
        run.summary.update(
            {"best_epoch": result["best_epoch"], SELECTION_METRIC: result["best_score"], **best_epoch_scores(result)}
        )
        return result
    finally:
        run.finish()


def run_from_spec(
    spec_path: str | Path, on_epoch: Callable[[int, dict[str, float]], None] | None = None
) -> dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("log_file"):
        logging.basicConfig(
            level=logging.INFO,
            filename=spec["log_file"],
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    fs = None
    if spec.get("use_drive"):
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, token="cache")
    kwargs = {
        "device": torch.device(spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")),
        "trial_dir": spec["trial_dir"],
        "stage_name": spec["stage_name"],
        "checkpoint": spec.get("checkpoint", str(ORIGINAL_CHECKPOINT)),
        "high_snr_levels": tuple(spec.get("high_snr_levels", (15.0, 20.0))),
        "max_samples": spec.get("max_samples"),
        "num_workers": spec.get("num_workers", 4),
        "smoke": spec.get("smoke", False),
        "fs": fs,
        "drive_dir": spec.get("drive_dir", DRIVE_SWEEP_DIR),
        "keep_history": bool(spec.get("keep_history", False)),
    }
    if on_epoch is not None and not spec.get("wandb_project"):
        kwargs["on_epoch"] = on_epoch
    if spec.get("dynamic_36l"):
        from scripts.dynamic_mix_36l import make_dynamic_36l_loaders

        kwargs["make_loaders"] = make_dynamic_36l_loaders(int(spec["dynamic_36l"].get("seed", 0)))
    elif spec.get("mixdataset_clean"):
        from scripts.mixdataset_speech import make_mixdataset_loaders

        kwargs["make_loaders"] = make_mixdataset_loaders(
            int(spec["mixdataset_clean"].get("seed", 0)), spec["mixdataset_clean"].get("train_snr")
        )
    elif spec.get("audioset_noise"):
        reference = spec["audioset_noise"]
        kwargs["make_loaders"] = make_audioset_noise_loaders(
            reference["offsets_dir"], reference["cache_dir"], float(reference["max_error"])
        )
    if spec.get("wandb_project"):
        return run_tracked_trial(
            spec["settings"],
            project=spec["wandb_project"],
            run_name=spec["run_name"],
            run_id=spec.get("run_id"),
            **kwargs,
        )
    return run_trial(spec["settings"], **kwargs)


def training_pid(pid_file: str | Path) -> int | None:
    path = Path(pid_file)
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
    except (ValueError, OSError):
        return None
    return pid if PROCESS_MARKER in command else None


def launch_training_process(spec: Mapping[str, Any], work_dir: str | Path) -> int:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    pid_file = work / "train.pid"
    running = training_pid(pid_file)
    if running is not None:
        return running
    spec_path = work / "spec.json"
    log_file = work / "train.log"
    spec_path.write_text(json.dumps({**spec, "log_file": str(log_file)}, indent=2), encoding="utf-8")
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from scripts.sweep_frcrn import {PROCESS_MARKER}; {PROCESS_MARKER}(sys.argv[1])"
    env = dict(os.environ, TQDM_DISABLE="1", WANDB_CONSOLE="off", WANDB_SILENT="true", PYTHONUNBUFFERED="1")
    with open(work / "stdout.log", "ab") as stdout:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(spec_path)],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def read_training_result(result_file: str | Path, work_dir: str | Path) -> dict[str, Any]:
    path = Path(result_file)
    if not path.is_file():
        tail = ""
        for name in ("train.log", "stdout.log"):
            log = Path(work_dir) / name
            if log.is_file():
                lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:]
                tail += f"--- {name}\n" + "\n".join(lines) + "\n"
        raise RuntimeError(f"training ended without {path}\n{tail}")
    return json.loads(path.read_text(encoding="utf-8"))


def make_wandb_trial(
    *,
    device: torch.device,
    root_dir: str | Path,
    checkpoint: str | Path = ORIGINAL_CHECKPOINT,
    high_snr_levels: Iterable[float] = (15.0, 20.0),
    max_samples: int | None = None,
    num_workers: int = 4,
    smoke: bool = False,
    fs: Any = None,
    drive_dir: str = DRIVE_SWEEP_DIR,
) -> Callable[[], None]:
    import wandb

    drive_dir = check_drive_dir(drive_dir)

    def trial() -> None:
        with wandb.init() as run:
            result = run_trial(
                dict(run.config),
                device=device,
                trial_dir=Path(root_dir) / run.id,
                checkpoint=checkpoint,
                high_snr_levels=high_snr_levels,
                max_samples=max_samples,
                num_workers=num_workers,
                smoke=smoke,
                on_epoch=lambda epoch, scores: run.log({"epoch": epoch, **scores}),
                stage_name=f"sweep-{run.id}",
                fs=fs,
                drive_dir=drive_dir,
            )
            run.summary.update(
                {
                    "best_epoch": result["best_epoch"],
                    SELECTION_METRIC: result["best_score"],
                    **best_epoch_scores(result),
                }
            )

    return trial


def validation_loader(
    batch_size: int = 32, seconds: float = CLIP_SECONDS, max_samples: int | None = None, num_workers: int = 4
) -> DataLoader:
    dataset = Mixture36Dataset(split="validation", duration=seconds, load_noise=True)
    return build_dataloader(subsample(dataset, max_samples), batch_size, False, num_workers)


@torch.no_grad()
def evaluate_by_snr(
    separator: FRCRNSeparator,
    loader: DataLoader,
    device: torch.device,
    high_snr_levels: Iterable[float] = (15.0, 20.0),
) -> dict[str, float]:
    separator.to(device).eval()
    speech, noise, noisy = (SnrBucketMeter(high_snr_levels) for _ in range(3))
    for batch in tqdm(loader, desc="evaluate by snr", unit="batch", leave=False, dynamic_ncols=True):
        mixture, noise_ref = batch["waveform"].to(device), batch["noise"].to(device)
        estimate = separator(mixture)
        clean = mixture - noise_ref
        speech.update(si_snr(clean, mixture - estimate), batch["snr_db"])
        noise.update(si_snr(noise_ref, estimate), batch["snr_db"])
        noisy.update(si_snr(clean, mixture), batch["snr_db"])
    result = {**speech.compute("speech_si_snr"), **noise.compute("noise_si_snr"), **noisy.compute("input_si_snr")}
    for level, value in speech.means().items():
        result[f"speech_si_snri_{level:g}"] = value - noisy.means()[level]
    return result


@torch.no_grad()
def perceptual_scores(
    separator: FRCRNSeparator,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = 20,
    high_snr_levels: Iterable[float] = (15.0, 20.0),
) -> dict[str, float]:
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError as error:
        raise ImportError("perceptual_scores needs the 'pesq' and 'pystoi' packages (see requirements.txt)") from error

    separator.to(device).eval()
    pesq_meter, stoi_meter = SnrBucketMeter(high_snr_levels), SnrBucketMeter(high_snr_levels)
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        mixture, noise_ref = batch["waveform"].to(device), batch["noise"].to(device)
        estimate = (mixture - separator(mixture)).cpu().numpy()
        clean = (mixture - noise_ref).cpu().numpy()
        for reference, enhanced, level in zip(clean, estimate, batch["snr_db"].numpy(), strict=True):
            pesq_meter.update([pesq(SAMPLE_RATE, reference, enhanced, "wb")], [level])
            stoi_meter.update([stoi(reference, enhanced, SAMPLE_RATE, extended=False)], [level])
    return {**pesq_meter.compute("pesq"), **stoi_meter.compute("stoi")}


def sweep_table(
    results: Mapping[str, Mapping[str, Any]], levels: Iterable[float] = (-5, 0, 5, 10, 15, 20)
) -> list[dict]:
    columns = [f"noise_si_snr_{float(level):g}" for level in levels]
    rows = []
    for name, scores in results.items():
        row = {"run": name, SELECTION_METRIC: scores.get(SELECTION_METRIC, np.nan)}
        row.update({column: scores.get(column, np.nan) for column in columns})
        rows.append(row)
    return sorted(rows, key=lambda row: (np.isnan(row[SELECTION_METRIC]), -np.nan_to_num(row[SELECTION_METRIC])))


__all__ = [
    "DRIVE_SWEEP_DIR",
    "ORIGINAL_CHECKPOINT",
    "SELECTION_METRIC",
    "best_epoch_scores",
    "build_trial_separator",
    "check_drive_dir",
    "evaluate_by_snr",
    "launch_training_process",
    "load_sweep_config",
    "make_wandb_trial",
    "perceptual_scores",
    "read_training_result",
    "run_from_spec",
    "run_tracked_trial",
    "run_trial",
    "sweep_table",
    "training_pid",
    "trial_train_config",
    "validation_loader",
]
