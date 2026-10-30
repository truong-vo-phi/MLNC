from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import build_student  # noqa: E402
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader  # noqa: E402
from src.metrics import snr_scores  # noqa: E402
from src.model.beat import load_fold_teacher  # noqa: E402
from src.model.student import CEDStudent  # noqa: E402
from src.utils.files import rms_normalize  # noqa: E402

CONFIG_PATH = ROOT / "config" / "kd_ced_tiny.yaml"
BEAT_CONFIG_PATH = ROOT / "config" / "beat.yaml"

DEFAULT_LOW_CKPT = Path("/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt")
DEFAULT_HIGH_CKPT = Path("/marimo/output/high_expert/last_best_checkpoint.pt")
DEFAULT_TEACHER_FOLD0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
DEFAULT_TEACHER_EMA = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")
DEFAULT_DATA_ROOT = Path("/marimo/dataset/mix-dataset-v2")
DEFAULT_CACHE_DIR = Path("/marimo/dataset/cache")


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return e / np.sum(e, axis=-1, keepdims=True)


def tta_forward(
    model: torch.nn.Module,
    wave: torch.Tensor,
    device: torch.device,
    shift_samples: int = 0,
    rms: float = 0.07925,
) -> torch.Tensor:
    if shift_samples != 0:
        if shift_samples > 0:
            w = F.pad(wave[:, shift_samples:], (0, shift_samples), mode="reflect")
        else:
            s = abs(shift_samples)
            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
    else:
        w = wave

    noise = rms_normalize(w, rms)
    with torch.autocast(device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
        logits = model(noise).logits
    return logits.float()


def run_tta_predict(
    model: torch.nn.Module,
    wave: torch.Tensor,
    device: torch.device,
    shift_delta: int = 800,
) -> torch.Tensor:
    l0 = tta_forward(model, wave, device, shift_samples=0)
    lp = tta_forward(model, wave, device, shift_samples=shift_delta)
    ln = tta_forward(model, wave, device, shift_samples=-shift_delta)
    return (l0 + 0.5 * lp + 0.5 * ln) / 2.0


class ChampionPipeline:
    def __init__(
        self,
        low_student: CEDStudent,
        high_student: CEDStudent,
        teacher_fold0: torch.nn.Module,
        teacher_ema: torch.nn.Module,
        device: torch.device,
        snr_threshold_db: float = 18.0,
        temp_student: float = 1.40,
        temp_teacher: float = 1.20,
    ) -> None:
        self.low_student = low_student.to(device).eval()
        self.high_student = high_student.to(device).eval()
        self.teacher_fold0 = teacher_fold0.to(device).eval()
        self.teacher_ema = teacher_ema.to(device).eval()
        self.device = device
        self.snr_threshold = snr_threshold_db
        self.temp_s = temp_student
        self.temp_t = temp_teacher

    @torch.no_grad()
    def predict_batch(self, wave: torch.Tensor, snrs_db: np.ndarray) -> np.ndarray:
        wave = wave.to(self.device, non_blocking=True)
        is_high_snr = snrs_db >= self.snr_threshold

        t0_logits = run_tta_predict(self.teacher_fold0, wave, self.device)
        tema_logits = run_tta_predict(self.teacher_ema, wave, self.device)

        p_t0 = softmax(t0_logits.cpu().numpy(), temp=self.temp_t)
        p_tema = softmax(tema_logits.cpu().numpy(), temp=self.temp_t)

        final_probs = np.zeros_like(p_t0)

        # 1. Low-Expert route (SNR < 18 dB)
        if not np.all(is_high_snr):
            low_mask = ~is_high_snr
            w_low = wave[low_mask]
            s_low_logits = run_tta_predict(self.low_student, w_low, self.device)
            p_s_low = softmax(s_low_logits.cpu().numpy(), temp=self.temp_s)

            final_probs[low_mask] = (
                0.30 * p_s_low + 0.45 * p_t0[low_mask] + 0.25 * p_tema[low_mask]
            )

        # 2. High-Expert route (SNR >= 18 dB)
        if np.any(is_high_snr):
            high_mask = is_high_snr
            w_high = wave[high_mask]
            s_high_logits = run_tta_predict(self.high_student, w_high, self.device)
            p_s_high = softmax(s_high_logits.cpu().numpy(), temp=self.temp_s)

            final_probs[high_mask] = (
                0.35 * p_s_high + 0.40 * p_tema[high_mask] + 0.25 * p_t0[high_mask]
            )

        return final_probs


def load_pipeline(
    config_path: Path,
    low_ckpt: Path,
    high_ckpt: Path,
    t_fold0_ckpt: Path,
    t_ema_ckpt: Path,
    device: torch.device,
) -> ChampionPipeline:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    low_student = build_student(cfg)
    low_student.load_pretrained(low_ckpt)

    high_student = build_student(cfg)
    high_data = torch.load(high_ckpt, map_location="cpu", weights_only=False)
    high_student.load_state_dict(high_data["model"])

    t_fold0, _ = load_fold_teacher(t_fold0_ckpt, BEAT_CONFIG_PATH, device)
    t_ema, _ = load_fold_teacher(t_ema_ckpt, BEAT_CONFIG_PATH, device)

    return ChampionPipeline(low_student, high_student, t_fold0, t_ema, device)


def evaluate_pipeline(
    pipeline: ChampionPipeline,
    data_root: Path,
    cache_dir: Path,
    split: str = "validation",
    batch_size: int = 64,
) -> dict[str, Any]:
    base = Mixture36Dataset(root=data_root, split=split, duration=4.0)
    cache_file = cache_dir / f"{split}.npy"
    done_file = cache_dir / f"{split}.done"
    cache_dataset = NoiseCacheDataset(base, cache_file, done_file)
    loader = build_dataloader(cache_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_snrs: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"Evaluating Champion Pipeline ({split})"):
        wave = batch["waveform"]
        snrs = batch["snr_db"].numpy()
        targets = batch["target"].argmax(dim=-1).numpy()

        probs = pipeline.predict_batch(wave, snrs)
        all_probs.append(probs)
        all_labels.append(targets)
        all_snrs.append(snrs)

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)

    return snr_scores(probs_arr, labels_arr, snrs_arr, 36)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Champion Pipeline")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--low-ckpt", type=Path, default=DEFAULT_LOW_CKPT)
    parser.add_argument("--high-ckpt", type=Path, default=DEFAULT_HIGH_CKPT)
    parser.add_argument("--teacher-fold0", type=Path, default=DEFAULT_TEACHER_FOLD0)
    parser.add_argument("--teacher-ema", type=Path, default=DEFAULT_TEACHER_EMA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("/marimo/output/kd-v5/champion_pipeline"))
    args = parser.parse_args()

    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")

    pipeline = load_pipeline(
        args.config,
        args.low_ckpt,
        args.high_ckpt,
        args.teacher_fold0,
        args.teacher_ema,
        dev,
    )

    scores = evaluate_pipeline(pipeline, args.data_root, args.cache_dir, args.split, args.batch_size)

    print("=============================================================================")
    print("CHAMPION PIPELINE EVALUATION SUMMARY:")
    print(f"Top-1 Accuracy:  {scores['top1_accuracy'] * 100:.2f}%")
    print(f"Top-3 Accuracy:  {scores['top3_accuracy'] * 100:.2f}%")
    print(f"Balanced Acc:    {scores['balanced_accuracy'] * 100:.2f}%")
    print(f"mAP (Macro):     {scores.get('mAP_macro', 0) * 100:.2f}%")
    print(f"mAP (Overall):   {scores['mAP'] * 100:.2f}%")
    print(f"Macro AUC:       {scores['macro_auc'] * 100:.2f}%")
    print("=============================================================================")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved evaluation metrics to: {summary_path}")


if __name__ == "__main__":
    main()
