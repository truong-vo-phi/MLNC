"""Đóng góp của nhánh noise và nhánh speech theo từng mức SNR (ablation hoán vị + Shapley hai nhóm).

Mô hình hai nhánh (đóng băng, không huấn luyện): mixture -> FRCRN -> nhiễu ước lượng n̂; tiếng nói ước lượng ŝ = mixture - n̂.
Mỗi nhánh qua BEATs pretrain -> embedding. Điểm lớp = tổng cosine tới centroid của từng nhánh (centroid tính từ tập train):
    score = cos(noise_emb, centroid_noise) + cos(speech_emb, centroid_speech)
Phá một nhánh = thay embedding của nó bằng embedding của clip khác cùng bin SNR (hoán vị, không tạo đầu vào ngoài phân phối).

    python -m scripts.branch_contribution --train-per-class 100 --test-per-class 180 --checkpoint best
    python -m scripts.branch_contribution --embeddings output/result/branch-contribution/embeddings-best.npz   # chỉ phân tích lại
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "output" / "result" / "branch-contribution"
CHECKPOINT_TARGET = {"best": "speech", "best_v2": "noise"}  # target FRCRN, đã xác định ở scripts.compare_frcrn_beats
NUM_CLASSES = 36

logger = logging.getLogger("branch_contribution")


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


# ---------- phân tích (numpy thuần, không cần mô hình) ----------


def class_scores(train: dict[str, np.ndarray], test: dict[str, np.ndarray], branch: str, num_classes: int) -> np.ndarray:
    """Cosine của embedding test tới centroid từng lớp, tính từ embedding train của cùng nhánh."""
    train_x = unit(train[f"{branch}_embedding"])
    centroids = unit(np.stack([train_x[train["label"] == c].mean(axis=0) for c in range(num_classes)]))
    return unit(test[f"{branch}_embedding"]) @ centroids.T


def permuted_correctness(
    noise_scores: np.ndarray,
    speech_scores: np.ndarray,
    labels: np.ndarray,
    bins: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Đúng/sai từng clip (trung bình trên `repeats` lần hoán vị) cho bốn điều kiện: full, no_noise, no_speech, none."""
    count = len(labels)
    result = {name: np.zeros(count) for name in ("full", "no_noise", "no_speech", "none")}
    for _ in range(repeats):
        shuffle_n, shuffle_s = np.arange(count), np.arange(count)
        for value in np.unique(bins):
            members = np.flatnonzero(bins == value)
            shuffle_n[members] = rng.permutation(members)
            shuffle_s[members] = rng.permutation(members)
        noise_kept, noise_broken = noise_scores, noise_scores[shuffle_n]
        speech_kept, speech_broken = speech_scores, speech_scores[shuffle_s]
        for name, (n, s) in {
            "full": (noise_kept, speech_kept),
            "no_noise": (noise_broken, speech_kept),
            "no_speech": (noise_kept, speech_broken),
            "none": (noise_broken, speech_broken),
        }.items():
            result[name] += ((n + s).argmax(axis=1) == labels) / repeats
    return result


def contributions(correct: dict[str, np.ndarray], index: np.ndarray) -> dict[str, float]:
    full, no_noise, no_speech, none = (correct[k][index].mean() for k in ("full", "no_noise", "no_speech", "none"))
    drop_noise, drop_speech = max(full - no_noise, 0.0), max(full - no_speech, 0.0)
    ablation_total = drop_noise + drop_speech + 1e-12
    shapley_noise = 0.5 * ((no_speech - none) + (full - no_noise))  # noise có mặt, speech bị phá / cả hai có mặt
    shapley_speech = 0.5 * ((no_noise - none) + (full - no_speech))
    shapley_total = abs(shapley_noise) + abs(shapley_speech) + 1e-12
    return {
        "acc_full": float(full),
        "acc_no_noise": float(no_noise),
        "acc_no_speech": float(no_speech),
        "acc_none": float(none),
        "ablation_noise_pct": float(100 * drop_noise / ablation_total),
        "ablation_speech_pct": float(100 * drop_speech / ablation_total),
        "shapley_noise": float(shapley_noise),
        "shapley_speech": float(shapley_speech),
        "shapley_noise_pct": float(100 * abs(shapley_noise) / shapley_total),
        "shapley_speech_pct": float(100 * abs(shapley_speech) / shapley_total),
    }


def analyze(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    repeats: int = 20,
    bootstrap: int = 200,
    seed: int = 0,
    num_classes: int = NUM_CLASSES,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    noise_scores = class_scores(train, test, "noise", num_classes)
    speech_scores = class_scores(train, test, "speech", num_classes)
    labels, snr = test["label"], test["snr_db"]
    correct = permuted_correctness(noise_scores, speech_scores, labels, snr, repeats, rng)
    report: dict[str, dict[str, Any]] = {}
    groups = {"all": np.arange(len(labels)), **{f"{int(v)}": np.flatnonzero(snr == v) for v in np.unique(snr)}}
    for name, index in groups.items():
        row: dict[str, Any] = {"n": int(len(index)), **contributions(correct, index)}
        row["acc_noise_only"] = float((noise_scores[index].argmax(axis=1) == labels[index]).mean())
        row["acc_speech_only"] = float((speech_scores[index].argmax(axis=1) == labels[index]).mean())
        draws = [contributions(correct, rng.choice(index, len(index))) for _ in range(bootstrap)]
        for key in ("shapley_noise_pct", "ablation_noise_pct"):
            low, high = np.percentile([d[key] for d in draws], [2.5, 97.5])
            row[f"{key}_ci"] = [float(low), float(high)]
        report[name] = row
    return report


def format_table(report: dict[str, dict[str, Any]]) -> str:
    header = f"{'SNR':>5} {'n':>5} {'full':>6} {'-noise':>7} {'-speech':>8} {'none':>6} {'abl N%':>7} {'abl S%':>7} {'shap N%':>8} {'[95% CI]':>15} {'shap S%':>8}"
    lines = [header]
    for name, r in report.items():
        low, high = r["shapley_noise_pct_ci"]
        lines.append(
            f"{name:>5} {r['n']:>5} {r['acc_full']:>6.3f} {r['acc_no_noise']:>7.3f} {r['acc_no_speech']:>8.3f} {r['acc_none']:>6.3f} "
            f"{r['ablation_noise_pct']:>7.1f} {r['ablation_speech_pct']:>7.1f} {r['shapley_noise_pct']:>8.1f} "
            f"{f'[{low:.0f},{high:.0f}]':>15} {r['shapley_speech_pct']:>8.1f}"
        )
    return "\n".join(lines)


def plot_report(report: dict[str, dict[str, Any]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [k for k in report if k != "all"]
    noise = [report[k]["shapley_noise_pct"] for k in snrs]
    speech = [report[k]["shapley_speech_pct"] for k in snrs]
    figure, axis = plt.subplots(figsize=(6, 3.6))
    axis.bar(snrs, noise, label="noise branch", color="#c0504d")
    axis.bar(snrs, speech, bottom=noise, label="speech branch", color="#4f81bd")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Shapley contribution (%)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------- tính embedding hai nhánh (cần mô hình) ----------


@torch.no_grad()
def embed_branches(loader: Any, beats: Any, separator: Any, device: torch.device) -> dict[str, np.ndarray]:
    from scripts.teacher_cache import beats_embedding, rms_normalize

    from scripts.compare_frcrn_beats import TARGET_RMS

    noise_emb, speech_emb, labels, snrs = [], [], [], []
    for batch in loader:
        mixture = batch["waveform"].to(device)
        noise_hat = separator(mixture)
        speech_hat = mixture - noise_hat
        noise_emb.append(beats_embedding(beats, rms_normalize(noise_hat, TARGET_RMS)).float().cpu().numpy())
        speech_emb.append(beats_embedding(beats, rms_normalize(speech_hat, TARGET_RMS)).float().cpu().numpy())
        labels.extend(batch["target"].argmax(dim=1).tolist())
        snrs.extend(batch["snr_db"].tolist())
    return {
        "noise_embedding": np.concatenate(noise_emb),
        "speech_embedding": np.concatenate(speech_emb),
        "label": np.asarray(labels),
        "snr_db": np.asarray(snrs),
    }


def compute_embeddings(args: argparse.Namespace) -> Path:
    from scripts.compare_frcrn_beats import (
        BEATS_CHECKPOINT,
        BEATS_CONFIG,
        CHECKPOINTS,
        load_separator,
        make_loader,
    )
    from scripts.download_dataset import ensure_dataset
    from scripts.load_dataset import MIXTURE_KEY
    from scripts.load_pretrain import download_pretrained
    from src.model.beat import BEATClassifier

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    ensure_dataset(MIXTURE_KEY, Path(args.download_root))
    train_loader = make_loader("train", args.train_per_class, args.batch_size, args.workers)
    test_loader = make_loader("test", args.test_per_class, args.batch_size, args.workers)
    logger.info("train clips %d, test clips %d", len(train_loader.dataset), len(test_loader.dataset))
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    beats.load_pretrained(BEATS_CHECKPOINT if BEATS_CHECKPOINT.is_file() else download_pretrained("beats"))
    beats = beats.freeze().to(device)
    separator = load_separator(CHECKPOINTS[args.checkpoint], CHECKPOINT_TARGET[args.checkpoint]).to(device)
    start = time.time()
    train = embed_branches(train_loader, beats, separator, device)
    test = embed_branches(test_loader, beats, separator, device)
    logger.info("embeddings done in %.0fs", time.time() - start)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"embeddings-{args.checkpoint}.npz"
    np.savez(path, **{f"train_{k}": v for k, v in train.items()}, **{f"test_{k}": v for k, v in test.items()})
    return path


def load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path) as data:
        train = {k.removeprefix("train_"): data[k] for k in data.files if k.startswith("train_")}
        test = {k.removeprefix("test_"): data[k] for k in data.files if k.startswith("test_")}
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, default=None, help="npz đã tính, bỏ qua bước chạy mô hình")
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINT_TARGET), default="best")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download-root", default=str(ROOT / "data"))
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    path = args.embeddings or compute_embeddings(args)
    train, test = load_embeddings(path)
    report = analyze(train, test, args.repeats, args.bootstrap)
    stem = path.stem.removeprefix("embeddings-")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"contribution-{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, RESULT_DIR / f"contribution-{stem}.png")
    print(format_table(report))
    logger.info("saved to %s", RESULT_DIR)


if __name__ == "__main__":
    main()
