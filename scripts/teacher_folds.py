"""Huấn luyện teacher BEATs theo 5 fold trên train (cross-fit), validation và test không dùng để huấn luyện hay chọn epoch.

    python -m scripts.teacher_folds --epochs 20 --patience 5

Fold chia theo noise_ytid (StratifiedGroupKFold, nhãn là lớp của clip). Mỗi fold: huấn luyện trên 4/5, chọn epoch theo
macro_f1 trên 1/5 còn lại (held-out), lưu best.pt, logits out-of-fold, logits validation và validation_clean.
Đầu ra: output/teacher-folds/{folds.csv, fold{k}/..., summary.json}. Fold đã có done.json được bỏ qua.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.finetune import BEAT_CONFIG, METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY, Mixture36Dataset, build_dataloader, subsample  # noqa: E402
from scripts.load_pretrain import download_pretrained  # noqa: E402
from src.metrics import ClassificationMetrics  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.train import BEATTrainConfig, BEATTrainer  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT_DIR = ROOT / "output" / "teacher-folds"
CLIP_SECONDS = 4.0
N_SPLITS = 5
NUM_CLASSES = 36


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_ids(path: Path, ids: Sequence[str]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"])
        writer.writerows([[sample_id] for sample_id in ids])
    return path


def label_of(row: dict[str, str]) -> int:
    return int(np.argmax(json.loads(row["multi_hot_36"])))


def assign_folds(rows: list[dict[str, str]], seed: int) -> dict[str, int]:
    labels = np.array([label_of(row) for row in rows])
    groups = [row["noise_ytid"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold = np.zeros(len(rows), dtype=int)
    for k, (_, held) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        fold[held] = k
    return {row["sample_id"]: int(f) for row, f in zip(rows, fold, strict=True)}


def write_folds(root: Path, assignment: dict[str, int], rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "folds.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "fold", "noise_ytid", "label"])
        for row in rows:
            writer.writerow([row["sample_id"], assignment[row["sample_id"]], row["noise_ytid"], label_of(row)])


@torch.no_grad()
def predict(model: BEATClassifier, loader, device: torch.device) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids, logits = [], []
    for batch in loader:
        logits.append(model(batch["waveform"].to(device)).logits.float().cpu().numpy())
        ids.extend(batch["sample_id"])
    return ids, np.concatenate(logits)


def save_logits(path: Path, ids: Sequence[str], logits: np.ndarray) -> None:
    np.savez(path, sample_id=np.array(ids), logits=logits)


def train_fold(
    k: int,
    fold_dir: Path,
    fold_csvs: dict[str, Path],
    config: BEATTrainConfig,
    device: torch.device,
    logger: logging.Logger,
    dataset_root: Path,
    max_samples: int | None,
    num_workers: int,
) -> dict[str, Any]:
    if (fold_dir / "done.json").is_file():
        logger.info("fold %d: finished earlier, skipping", k)
        return json.loads((fold_dir / "done.json").read_text(encoding="utf-8"))

    fold_dir.mkdir(parents=True, exist_ok=True)

    def loader(split: str, size: int, shuffle: bool, ids_file: Path | None = None):
        dataset = Mixture36Dataset(
            dataset_root, split=split, duration=CLIP_SECONDS, load_noise=False, sample_ids_file=ids_file
        )
        return build_dataloader(subsample(dataset, max_samples), size, shuffle, num_workers)

    train_loader = loader("train", config.batch_size, True, fold_csvs["train"])
    held_loader = loader("train", config.eval_batch_size, False, fold_csvs["held"])
    validation_loader = loader("validation", config.eval_batch_size, False)
    clean_loader = loader("validation", config.eval_batch_size, False, fold_csvs["clean"])

    model = BEATClassifier.from_yaml(BEAT_CONFIG)
    report = model.load_pretrained(download_pretrained("beats"))
    logger.info("fold %d: pretrain loaded, missing %d, unexpected %d", k, len(report.missing), len(report.unexpected))
    model.to(device)

    checkpoint = fold_dir / "best.pt"
    resume = fold_dir / "resume.pt"
    trainer = BEATTrainer(model, config, device, logger, f"fold{k}", checkpoint, None, resume)
    if resume.is_file():
        trainer.load_resume(resume)
    fit = trainer.fit(train_loader, held_loader)
    trainer._in_fit = False

    held_ids, held_logits = predict(model, held_loader, device)
    save_logits(fold_dir / "oof_held.npz", held_ids, held_logits)
    val_ids, val_logits = predict(model, validation_loader, device)
    save_logits(fold_dir / "validation.npz", val_ids, val_logits)
    clean_ids, clean_logits = predict(model, clean_loader, device)
    save_logits(fold_dir / "validation_clean.npz", clean_ids, clean_logits)

    done = {
        "fold": k,
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        "best_macro_f1_held": fit.best_score,
        "held_n": len(held_ids),
    }
    (fold_dir / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    logger.info("fold %d: best epoch %d, held macro_f1 %.4f", k, fit.best_epoch, fit.best_score)
    return done


def ensemble_metrics(npz_paths: list[Path], labels: dict[str, int], num_classes: int) -> dict[str, float]:
    probs: dict[str, np.ndarray] = {}
    for path in npz_paths:
        data = np.load(path)
        softmax = torch.softmax(torch.from_numpy(data["logits"]), dim=-1).numpy()
        for sample_id, p in zip(data["sample_id"], softmax, strict=True):
            probs[str(sample_id)] = probs.get(str(sample_id), 0) + p / len(npz_paths)
    ids = sorted(probs)
    metrics = ClassificationMetrics(num_classes=num_classes)
    metrics.update(np.array([labels[i] for i in ids]), np.stack([probs[i] for i in ids]))
    return metrics.compute().to_dict()


def out_of_fold_metrics(fold_dirs: list[Path], labels: dict[str, int], num_classes: int) -> dict[str, float]:
    ids, logits = [], []
    for fold_dir in fold_dirs:
        data = np.load(fold_dir / "oof_held.npz")
        ids.extend(str(i) for i in data["sample_id"])
        logits.append(data["logits"])
    probs = torch.softmax(torch.from_numpy(np.concatenate(logits)), dim=-1).numpy()
    metrics = ClassificationMetrics(num_classes=num_classes)
    metrics.update(np.array([labels[i] for i in ids]), probs)
    return metrics.compute().to_dict()


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 64 mẫu, ghi vào output/test/teacher-folds-smoke")
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    args = parser.parse_args(argv)

    out_dir = OUT_DIR
    max_samples, workers = None, args.workers
    config = replace(
        BEATTrainConfig.from_yaml(BEAT_CONFIG), epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
    )
    if args.smoke:
        out_dir = ROOT / "output" / "test" / "teacher-folds-smoke"
        config = replace(config, epochs=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
        max_samples, workers = SMOKE_SAMPLES, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("teacher_folds", out_dir / "teacher_folds.log")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = ensure_dataset(MIXTURE_KEY, args.download_root)
    train_rows = read_rows(dataset_root / "train" / "manifest.csv")
    clean_ids = [row["sample_id"] for row in read_rows(dataset_root / "validation_clean.csv")]
    labels = {row["sample_id"]: label_of(row) for row in train_rows}
    assignment = assign_folds(train_rows, args.seed)
    write_folds(dataset_root, assignment, train_rows, out_dir)
    logger.info("pool %d clip, folds %s", len(train_rows), np.bincount(list(assignment.values())).tolist())

    validation_clean_file = write_ids(out_dir / "validation_clean_ids.csv", clean_ids)
    summary: dict[str, Any] = {"folds": []}
    fold_dirs = []
    for k in args.folds:
        held = [row["sample_id"] for row in train_rows if assignment[row["sample_id"]] == k]
        train = [row["sample_id"] for row in train_rows if assignment[row["sample_id"]] != k]
        fold_csvs = {
            "held": write_ids(out_dir / f"fold{k}_held.csv", held),
            "train": write_ids(out_dir / f"fold{k}_train.csv", train),
            "clean": validation_clean_file,
        }
        fold_dir = out_dir / f"fold{k}"
        fold_dirs.append(fold_dir)
        summary["folds"].append(
            train_fold(k, fold_dir, fold_csvs, config, device, logger, dataset_root, max_samples, workers)
        )
        torch.cuda.empty_cache()

    done_dirs = [out_dir / f"fold{k}" for k in args.folds if (out_dir / f"fold{k}" / "done.json").is_file()]
    if len(done_dirs) == N_SPLITS:
        summary["out_of_fold_pool"] = out_of_fold_metrics(done_dirs, labels, NUM_CLASSES)
        summary["validation_ensemble"] = ensemble_metrics(
            [d / "validation.npz" for d in done_dirs], validation_labels(dataset_root), NUM_CLASSES
        )
        summary["validation_clean_ensemble"] = ensemble_metrics(
            [d / "validation_clean.npz" for d in done_dirs], validation_labels(dataset_root), 36
        )
        logger.info("out-of-fold %s", {k: round(summary["out_of_fold_pool"][k], 4) for k in METRIC_NAMES})
        logger.info("validation ensemble %s", {k: round(summary["validation_ensemble"][k], 4) for k in METRIC_NAMES})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validation_labels(dataset_root: Path) -> dict[str, int]:
    return {row["sample_id"]: label_of(row) for row in read_rows(dataset_root / "validation" / "manifest.csv")}


if __name__ == "__main__":
    main()
