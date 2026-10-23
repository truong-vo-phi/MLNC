"""So sánh trực tiếp từng loại tín hiệu đưa vào BEATs pretrain đóng băng (không huấn luyện) theo SNR.

Năm đầu vào: tiếng nói sạch thật, tiếng nói ước lượng (mixture - nhiễu ước lượng), nhiễu thật, nhiễu ước lượng (FRCRN), mixture.
Phân loại: cosine tới centroid lớp (centroid từ embedding train của cùng loại đầu vào). Kèm đo nhiễu còn sót trong tiếng nói
ước lượng bằng SI-SNR (dB) so với nhiễu thật và so với tiếng nói sạch.

    python -m scripts.input_comparison --train-per-class 100 --test-per-class 180
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "output" / "result" / "input-comparison"
INPUTS = ("clean_speech", "speech_hat", "noise_true", "noise_hat", "mixture")
NUM_CLASSES = 36
logger = logging.getLogger("input_comparison")


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def centroid_accuracy(train: dict[str, np.ndarray], test: dict[str, np.ndarray], name: str) -> dict[str, float]:
    train_x = unit(train[name])
    centroids = unit(np.stack([train_x[train["label"] == c].mean(axis=0) for c in range(NUM_CLASSES)]))
    scores = unit(test[name]) @ centroids.T
    label, snr = test["label"], test["snr_db"]
    top1 = scores.argmax(axis=1) == label
    top3 = (np.argsort(-scores, axis=1)[:, :3] == label[:, None]).any(axis=1)
    report = {"top1_all": float(top1.mean()), "top3_all": float(top3.mean())}
    for value in np.unique(snr):
        report[f"top1_snr_{int(value)}"] = float(top1[snr == value].mean())
    return report


def residual_report(test: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """SI-SNR (dB) trung bình theo SNR của tiếng nói ước lượng so với nhiễu thật và với tiếng nói sạch."""
    report: dict[str, dict[str, float]] = {}
    for key in ("sisnr_speechhat_vs_noise", "sisnr_speechhat_vs_clean"):
        report[key] = {str(int(v)): float(test[key][test["snr_db"] == v].mean()) for v in np.unique(test["snr_db"])}
    return report


@torch.no_grad()
def embed(loader: Any, beats: Any, separator: Any, device: torch.device) -> dict[str, np.ndarray]:
    from scripts.compare_frcrn_beats import TARGET_RMS
    from scripts.teacher_cache import beats_embedding, rms_normalize
    from src.loss import si_snr

    out: dict[str, list] = {name: [] for name in INPUTS}
    labels, snrs, leak_noise, leak_clean = [], [], [], []
    for batch in loader:
        mixture, noise, clean = (batch[k].to(device) for k in ("waveform", "noise", "clean"))
        noise_hat = separator(mixture)
        signals = {
            "clean_speech": clean,
            "speech_hat": mixture - noise_hat,
            "noise_true": noise,
            "noise_hat": noise_hat,
            "mixture": mixture,
        }
        for name, signal in signals.items():
            out[name].append(beats_embedding(beats, rms_normalize(signal, TARGET_RMS)).float().cpu().numpy())
        leak_noise.extend(si_snr(noise, signals["speech_hat"]).tolist())
        leak_clean.extend(si_snr(clean, signals["speech_hat"]).tolist())
        labels.extend(batch["target"].argmax(dim=1).tolist())
        snrs.extend(batch["snr_db"].tolist())
    return {
        **{name: np.concatenate(chunks) for name, chunks in out.items()},
        "label": np.asarray(labels),
        "snr_db": np.asarray(snrs),
        "sisnr_speechhat_vs_noise": np.asarray(leak_noise, dtype=np.float32),
        "sisnr_speechhat_vs_clean": np.asarray(leak_clean, dtype=np.float32),
    }


def make_loader(split: str, per_class: int, batch_size: int, workers: int) -> DataLoader:
    from scripts.compare_frcrn_beats import stratified_indices
    from scripts.load_dataset import Mixture36Dataset

    dataset = Mixture36Dataset(split=split, duration=4.0, load_noise=True, load_clean=True)
    subset = Subset(dataset, stratified_indices(dataset, per_class))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=workers)


def plot_report(report: dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = sorted({int(k.split("_")[-1]) for r in report["accuracy"].values() for k in r if k.startswith("top1_snr_")})
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for name in INPUTS:
        axis.plot(snrs, [report["accuracy"][name][f"top1_snr_{s}"] for s in snrs], marker="o", label=name)
    axis.axhline(1 / NUM_CLASSES, color="grey", linestyle=":", label="chance")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("top-1 accuracy")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def format_table(report: dict[str, Any]) -> str:
    snrs = sorted({int(k.split("_")[-1]) for r in report["accuracy"].values() for k in r if k.startswith("top1_snr_")})
    lines = [f"{'input':<14}{'all':>7}" + "".join(f"{s:>7}" for s in snrs)]
    for name in INPUTS:
        row = report["accuracy"][name]
        lines.append(f"{name:<14}{row['top1_all']:>7.3f}" + "".join(f"{row[f'top1_snr_{s}']:>7.3f}" for s in snrs))
    for key, values in report["residual"].items():
        lines.append(f"{key:<26}" + " ".join(f"{s}dB:{values[str(s)]:.1f}" for s in snrs))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download-root", default=str(ROOT / "data"))
    parser.add_argument("--checkpoint", choices=("best", "best_v2"), default="best")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from scripts.compare_frcrn_beats import BEATS_CHECKPOINT, BEATS_CONFIG, CHECKPOINTS, load_separator
    from scripts.download_dataset import ensure_dataset
    from scripts.load_dataset import MIXTURE_KEY
    from scripts.load_pretrain import download_pretrained
    from src.model.beat import BEATClassifier

    device = torch.device(args.device)
    ensure_dataset(MIXTURE_KEY, Path(args.download_root))
    train_loader = make_loader("train", args.train_per_class, args.batch_size, args.workers)
    test_loader = make_loader("test", args.test_per_class, args.batch_size, args.workers)
    logger.info("train clips %d, test clips %d", len(train_loader.dataset), len(test_loader.dataset))
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    beats.load_pretrained(BEATS_CHECKPOINT if BEATS_CHECKPOINT.is_file() else download_pretrained("beats"))
    beats = beats.freeze().to(device)
    target = {"best": "speech", "best_v2": "noise"}[args.checkpoint]
    separator = load_separator(CHECKPOINTS[args.checkpoint], target).to(device)
    start = time.time()
    train = embed(train_loader, beats, separator, device)
    test = embed(test_loader, beats, separator, device)
    logger.info("embeddings done in %.0fs", time.time() - start)

    report = {
        "args": vars(args),
        "accuracy": {name: centroid_accuracy(train, test, name) for name in INPUTS},
        "residual": residual_report(test),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"input-comparison-{args.checkpoint}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(RESULT_DIR / f"embeddings-{args.checkpoint}.npz", **{f"train_{k}": v for k, v in train.items()}, **{f"test_{k}": v for k, v in test.items()})
    plot_report(report, RESULT_DIR / f"input-comparison-{args.checkpoint}.png")
    print(format_table(report))
    logger.info("saved to %s", RESULT_DIR)


if __name__ == "__main__":
    main()
