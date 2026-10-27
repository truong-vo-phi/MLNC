"""So sánh hai checkpoint FRCRN (best.pt, best_v2.pt) qua BEATs pretrain đóng băng, không fine-tune, không huấn luyện.

Luồng: mixture -> FRCRN (đầu ra là nhiễu ước lượng) -> chuẩn hoá RMS -> BEATs iter3+ AS2M -> embedding (trung bình token).
Phân loại không có tham số học: centroid của từng lớp tính từ embedding tập train (con), gán nhãn theo cosine gần nhất.
Kèm hai mốc: BEATs trên mixture thô (không FRCRN) và BEATs trên nhiễu sạch (trần).

    python -m scripts.compare_frcrn_beats --train-per-class 20 --test-per-class 20
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune import METRIC_NAMES  # noqa: E402
from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY, Mixture36Dataset  # noqa: E402
from scripts.load_pretrain import download_pretrained  # noqa: E402
from scripts.teacher_cache import beats_embedding, rms_normalize  # noqa: E402
from src.loss import si_snr  # noqa: E402
from src.metrics import ClassificationMetrics  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402

FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"
BEATS_CONFIG = ROOT / "config" / "beat.yaml"
BEATS_CHECKPOINT = ROOT / "pretrained" / "BEATs_iter3_plus_AS2M.pt"
CHECKPOINTS = {"best": ROOT / "pretrained" / "FRCRN" / "best.pt", "best_v2": ROOT / "pretrained" / "FRCRN" / "best_v2.pt"}
TARGET_RMS = 0.07925  # trung vị RMS nhiễu AudioSet train, lấy từ config/reviewkd.yaml
RESULT_DIR = ROOT / "output" / "result" / "frcrn-compare"
TARGETS = ("speech", "noise")

logger = logging.getLogger("compare_frcrn_beats")


def stratified_indices(dataset: Mixture36Dataset, per_class: int) -> list[int]:
    by_class: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        by_class[json.loads(row["multi_hot_36"]).index(1)].append(index)
    chosen: list[int] = []
    for label in sorted(by_class):
        rows = by_class[label]
        picks = np.linspace(0, len(rows) - 1, min(per_class, len(rows))).round().astype(int)
        chosen.extend(rows[i] for i in picks)
    return chosen


def make_loader(split: str, per_class: int, batch_size: int, workers: int) -> DataLoader:
    dataset = Mixture36Dataset(split=split, duration=4.0, load_noise=True)
    subset = Subset(dataset, stratified_indices(dataset, per_class))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=workers)


def load_separator(path: Path, target: str) -> FRCRNSeparator:
    separator = FRCRNSeparator.from_yaml(FRCRN_CONFIG)
    separator.config = type(separator.config)(**{**separator.config.__dict__, "target": target})
    separator.load_pretrained(path)
    return separator.freeze()


@torch.no_grad()
def pick_target(path: Path, loader: DataLoader, probe_batches: int, device: torch.device) -> tuple[str, dict[str, float]]:
    """Checkpoint không lưu `target`: thử cả hai và chọn cái cho SI-SNR nhiễu cao hơn."""
    scores: dict[str, float] = {}
    for target in TARGETS:
        separator = load_separator(path, target).to(device)
        values: list[float] = []
        for batch_index, batch in enumerate(loader):
            if batch_index >= probe_batches:
                break
            values.extend(si_snr(batch["noise"].to(device), separator(batch["waveform"].to(device))).tolist())
        scores[target] = float(np.mean(values))
    return max(scores, key=scores.__getitem__), scores


@torch.no_grad()
def embed_split(
    loader: DataLoader,
    beats: BEATClassifier,
    separator: FRCRNSeparator | None,
    device: torch.device,
    use_noise_input: bool = False,
) -> dict[str, np.ndarray]:
    embeddings, labels, snrs, sisnr = [], [], [], []
    for batch in loader:
        mixture = (batch["noise"] if use_noise_input else batch["waveform"]).to(device)
        estimate = separator(mixture) if separator is not None else mixture
        if separator is not None:
            sisnr.extend(si_snr(batch["noise"].to(device), estimate).tolist())
        embeddings.append(beats_embedding(beats, rms_normalize(estimate, TARGET_RMS)).float().cpu().numpy())
        labels.extend(batch["target"].argmax(dim=1).tolist())
        snrs.extend(batch["snr_db"].tolist())
    return {
        "embedding": np.concatenate(embeddings),
        "label": np.asarray(labels),
        "snr_db": np.asarray(snrs),
        "sisnr": np.asarray(sisnr, dtype=np.float32),
    }


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def centroid_scores(train: dict[str, np.ndarray], test: dict[str, np.ndarray], num_classes: int) -> dict[str, float]:
    train_x, test_x = unit(train["embedding"]), unit(test["embedding"])
    centroids = unit(np.stack([train_x[train["label"] == c].mean(axis=0) for c in range(num_classes)]))
    scores = test_x @ centroids.T  # cosine; chỉ thứ hạng được dùng bởi top-k, mAP, AUC
    label, snr = test["label"], test["snr_db"]
    metrics = ClassificationMetrics(num_classes=num_classes).update(label, scores).compute().to_dict()
    report = {name: float(metrics[name]) for name in METRIC_NAMES}
    correct = scores.argmax(axis=1) == label
    for value in np.unique(snr):
        report[f"acc_snr_{int(value)}"] = float(correct[snr == value].mean())
    if test["sisnr"].size:
        report["noise_sisnr_db"] = float(test["sisnr"].mean())
        for value in np.unique(snr):
            report[f"sisnr_snr_{int(value)}"] = float(test["sisnr"][snr == value].mean())
    return report


def timed(name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    start = time.time()
    result = function(*args, **kwargs)
    logger.info("%s done in %.0fs", name, time.time() - start)
    return result


def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    ensure_dataset(MIXTURE_KEY, Path(args.download_root))
    train_loader = make_loader("train", args.train_per_class, args.batch_size, args.workers)
    test_loader = make_loader("test", args.test_per_class, args.batch_size, args.workers)
    logger.info("train clips %d, test clips %d", len(train_loader.dataset), len(test_loader.dataset))
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    beats.load_pretrained(BEATS_CHECKPOINT if BEATS_CHECKPOINT.is_file() else download_pretrained("beats"))
    beats = beats.freeze().to(device)
    results: dict[str, dict[str, float]] = {}
    systems: dict[str, tuple[FRCRNSeparator | None, bool]] = {"mixture_only (không FRCRN)": (None, False), "noise_oracle (nhiễu sạch)": (None, True)}
    for name, path in CHECKPOINTS.items():
        target, scores = pick_target(path, test_loader, args.probe_batches, device)
        logger.info("%s: target=%s, noise SI-SNR theo target %s", name, target, scores)
        systems[f"frcrn_{name} (target={target})"] = (load_separator(path, target).to(device), False)
    for name, (separator, oracle) in systems.items():
        train = timed(f"{name} train", embed_split, train_loader, beats, separator, device, oracle)
        test = timed(f"{name} test", embed_split, test_loader, beats, separator, device, oracle)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(RESULT_DIR / f"embeddings-{name.split(' ')[0]}.npz", **{f"train_{k}": v for k, v in train.items()}, **{f"test_{k}": v for k, v in test.items()})
        results[name] = centroid_scores(train, test, len(test_loader.dataset.dataset.class_names))
        logger.info("%s -> %s", name, {k: round(v, 4) for k, v in results[name].items()})
    return results


def write_results(results: dict[str, dict[str, float]], args: argparse.Namespace) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    columns = sorted({key for row in results.values() for key in row}, key=lambda k: (k not in METRIC_NAMES, METRIC_NAMES.index(k) if k in METRIC_NAMES else 0, k))
    with open(RESULT_DIR / f"compare-{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", *columns])
        for name, row in results.items():
            writer.writerow([name, *(f"{row.get(c, float('nan')):.4f}" for c in columns)])
    path = RESULT_DIR / f"compare-{stamp}.json"
    path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-batches", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download-root", default=str(ROOT / "data"))
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    path = write_results(run(args), args)
    logger.info("saved %s", path)


if __name__ == "__main__":
    main()
