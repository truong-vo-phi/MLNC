"""
Comprehensive 9-Model Evaluation Benchmark Script for Review-3.
Evaluates 9 acoustic noise classification models under two conditions:
  1. No FRCRN: Raw clean mixture audio
  2. Mix + FRCRN: Noise extracted after FRCRN speech enhancement mask with RMS norm (0.07925)

Metrics:
  - acc (Top-1 Accuracy)
  - precision (Macro Precision)
  - recall (Macro Recall)
  - macro_f1 (Macro F1)
  - micro_f1 (Micro F1)
  - mAP (Mean Average Precision)
  - Detailed per-SNR breakdown (-5, 0, 5, 10, 15, 20 dB)

Models (Total 9):
  - panns_cnn10         (6,050,852 params)
  - ced_tiny            (5,408,420 params)
  - ced_mini            (9,570,468 params)
  - ced_small           (21,433,508 params)
  - efficientat_mn10    (4,247,860 params)
  - vit_small           (21,679,524 params)
  - mobilevit_s         (4,960,708 params)
  - mobilenet_v3_large  (4,248,148 params)
  - beats12             (90,339,476 params)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# Setup project roots
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CE_V2_ROOT = PACKAGE_ROOT / "ce_v2"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(CE_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(CE_V2_ROOT))

# Import model builders from ce_v2 noise pipeline
from noise_pipeline.ce9_models import build_model, MODEL_NAMES, beats_checkpoint

CANONICAL_9_MODELS = [
    "panns_cnn10",
    "ced_tiny",
    "ced_mini",
    "ced_small",
    "efficientat_mn10",
    "vit_small",
    "mobilevit_s",
    "mobilenet_v3_large",
    "beats12",
]

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
METRIC_KEYS = ["acc", "precision", "recall", "macro_f1", "micro_f1", "mAP"]


def compute_metrics(
    logits_or_probs: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 36,
) -> Dict[str, float]:
    """Computes Top-1 acc, macro precision, recall, macro-F1, micro-F1, and mAP."""
    preds_raw = np.asarray(logits_or_probs)
    targets_raw = np.asarray(targets)

    if targets_raw.ndim == 2:
        targets_idx = np.argmax(targets_raw, axis=-1)
    else:
        targets_idx = targets_raw.astype(np.int64)

    # Convert logits to probabilities if needed
    if preds_raw.ndim == 2 and (np.any(preds_raw < 0) or np.any(preds_raw > 1.0)):
        exps = np.exp(preds_raw - np.max(preds_raw, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
    else:
        probs = preds_raw

    pred_idx = np.argmax(probs, axis=-1)
    acc = float(np.mean(pred_idx == targets_idx)) if len(targets_idx) > 0 else 0.0

    p, r, f, _ = precision_recall_fscore_support(
        targets_idx, pred_idx, labels=np.arange(num_classes), average="macro", zero_division=0
    )
    _, _, micro, _ = precision_recall_fscore_support(
        targets_idx, pred_idx, average="micro", zero_division=0
    )

    aps = []
    for c in range(num_classes):
        binary_targets = (targets_idx == c)
        if np.any(binary_targets):
            try:
                ap = average_precision_score(binary_targets, probs[:, c])
                aps.append(ap)
            except Exception:
                pass
    mAP = float(np.mean(aps)) if aps else 0.0

    return {
        "acc": acc,
        "precision": float(p),
        "recall": float(r),
        "macro_f1": float(f),
        "micro_f1": float(micro),
        "mAP": mAP,
    }


def compute_snr_breakdown(
    logits: np.ndarray,
    targets: np.ndarray,
    snrs: np.ndarray,
    num_classes: int = 36,
) -> Dict[str, Dict[str, float]]:
    """Calculates metrics separated by individual SNR values and overall."""
    breakdown = {}
    for snr_val in SNR_LEVELS:
        mask = np.isclose(snrs, float(snr_val), atol=0.5)
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            sub_m = compute_metrics(logits[mask], targets[mask], num_classes=num_classes)
            sub_m["n_samples"] = n_samples
            breakdown[f"{snr_val}dB"] = sub_m
        else:
            breakdown[f"{snr_val}dB"] = {k: 0.0 for k in METRIC_KEYS}
            breakdown[f"{snr_val}dB"]["n_samples"] = 0

    overall = compute_metrics(logits, targets, num_classes=num_classes)
    overall["n_samples"] = len(targets)
    breakdown["overall"] = overall
    return breakdown


class RawMixtureDataset(torch.utils.data.Dataset):
    """Loads clean mixture raw waveforms without FRCRN masking."""

    def __init__(
        self,
        data_root: Path,
        split: str = "eval",
        target_len: int = 64000,
        max_samples: Optional[int] = None,
    ):
        self.target_len = target_len
        manifest_file = data_root / split / "manifest.csv"
        if not manifest_file.is_file():
            # Alternative layout: data_root / "manifest.csv"
            manifest_file = data_root / "manifest.csv"

        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest not found at: {manifest_file}")

        self.records = []
        with open(manifest_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)
                if max_samples and len(self.records) >= max_samples:
                    break

        self.audio_dir = data_root / split / "audio"
        if not self.audio_dir.is_dir():
            self.audio_dir = data_root / split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        row = self.records[idx]
        wav_name = row.get("sample_id", "")
        if not wav_name.endswith(".wav"):
            wav_name = f"{wav_name}.wav"
        wav_path = self.audio_dir / wav_name

        try:
            import soundfile as sf
            wav, sr = sf.read(str(wav_path), dtype="float32")
            if sr != 16000:
                import torchaudio.transforms as T
                resampler = T.Resample(sr, 16000)
                wav = resampler(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        except Exception:
            wav = np.zeros(self.target_len, dtype=np.float32)

        if len(wav) < self.target_len:
            pad = np.zeros(self.target_len - len(wav), dtype=np.float32)
            wav = np.concatenate([wav, pad])
        elif len(wav) > self.target_len:
            wav = wav[: self.target_len]

        targets = json.loads(row["multi_hot_36"])
        label = int(np.argmax(targets))
        snr = float(row.get("target_snr_db", 0.0))
        return torch.from_numpy(wav), label, snr


class MaskedNoiseCacheDataset(torch.utils.data.Dataset):
    """Loads FRCRN-masked noise waveforms with target RMS normalization."""

    def __init__(
        self,
        npy_path: Path,
        manifest_path: Path,
        target_len: int = 64000,
        target_rms: float = 0.07925,
        apply_rms_norm: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.target_len = target_len
        self.target_rms = target_rms
        self.apply_rms_norm = apply_rms_norm
        self.array = np.load(str(npy_path), mmap_mode="r")

        self.records = []
        with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)
                if max_samples and len(self.records) >= max_samples:
                    break

        self.n_samples = min(len(self.array), len(self.records))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        raw_wave = np.array(self.array[idx], copy=True, dtype=np.float32)

        # Pad / Crop to 64000
        if len(raw_wave) < self.target_len:
            pad = np.zeros(self.target_len - len(raw_wave), dtype=np.float32)
            wave = np.concatenate([raw_wave, pad])
        elif len(raw_wave) > self.target_len:
            wave = raw_wave[: self.target_len]
        else:
            wave = raw_wave

        # RMS Normalization
        if self.apply_rms_norm:
            rms = np.sqrt(np.mean(wave**2) + 1e-8)
            wave = wave * (self.target_rms / (rms + 1e-8))

        row = self.records[idx]
        targets = json.loads(row["multi_hot_36"])
        label = int(np.argmax(targets))
        snr = float(row.get("target_snr_db", 0.0))
        return torch.from_numpy(wave), label, snr


def run_evaluation_for_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward inference across dataset and returns predictions, targets, and snrs."""
    model.eval()
    all_logits = []
    all_targets = []
    all_snrs = []

    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch_wave, batch_target, batch_snr in dataloader:
            batch_wave = batch_wave.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_wave)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(batch_target.numpy())
            all_snrs.append(batch_snr.numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_snrs, axis=0),
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Evaluate 9 Audio Models (No FRCRN vs Mix + FRCRN)")
    parser.add_argument("--config", type=str, default=str(PACKAGE_ROOT / "config" / "config.yaml"))
    parser.add_argument("--models-config", type=str, default=str(PACKAGE_ROOT / "config" / "models.yaml"))
    parser.add_argument("--data-root", type=str, default="/datasets/mix-frcrn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PACKAGE_ROOT / "logs"))
    args = parser.parse_args()

    # Load configuration
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("REVIEW-3: 9-MODEL BENCHMARK EVALUATION (No FRCRN vs Mix + FRCRN)")
    print(f"Device: {device} | Data Root: {data_root}")
    print("=" * 88)

    # Assets directory for pretrained weights
    assets_dir = CE_V2_ROOT / "benchmark_results" / "ce9_clean_mixture_20261102" / "pretrained"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Candidate evaluation data paths
    manifest_candidates = [
        data_root / "mix-dataset-v2" / "eval" / "manifest.csv",
        data_root / "eval" / "manifest.csv",
        data_root / "manifest.csv",
        Path("/marimo/datasets/mix-dataset-v2/eval/manifest.csv"),
    ]
    manifest_path = next((p for p in manifest_candidates if p.is_file()), None)

    frcrn_cache_candidates = [
        data_root / "validation.npy",
        data_root / "mix-frcrn" / "validation.npy",
        data_root / "raw_mixture_cache" / "eval_mixture.npy",
        Path("/marimo/datasets/validation.npy"),
    ]
    frcrn_cache_path = next((p for p in frcrn_cache_candidates if p.is_file()), None)

    results_table = []

    for model_name in CANONICAL_9_MODELS:
        print(f"\n[+] Building model adapter: {model_name}...")
        try:
            model = build_model(model_name, assets=assets_dir, classes=36).to(device)
            p_count = count_params(model)
            print(f"    Parameters: {p_count:,}")
        except Exception as e:
            print(f"    [Error building {model_name}]: {e}")
            continue

        # Check for fine-tuned weights
        ckpt_candidates = [
            PACKAGE_ROOT / "checkpoints" / model_name / "best_model.pt",
            CE_V2_ROOT / "benchmark_results" / "ce9_clean_mixture_20261102" / model_name / "checkpoints" / "best.pt",
            PACKAGE_ROOT / "checkpoints" / f"{model_name}.pt",
        ]
        ckpt_path = next((p for p in ckpt_candidates if p.is_file()), None)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state.get("model_state_dict", state.get("model", state)), strict=False)
            print(f"    Loaded checkpoint: {ckpt_path.name}")
        else:
            print("    Running with pretrained / initialized weights")

        # 1. Evaluate No FRCRN (Clean raw mixture)
        if manifest_path:
            try:
                ds_no_frcrn = RawMixtureDataset(
                    data_root=manifest_path.parent.parent,
                    split=manifest_path.parent.name,
                    max_samples=args.max_samples,
                )
                loader_no_frcrn = torch.utils.data.DataLoader(
                    ds_no_frcrn, batch_size=args.batch_size, shuffle=False, num_workers=2
                )
                logits, targets, snrs = run_evaluation_for_model(model, loader_no_frcrn, device)
                res_no_frcrn = compute_metrics(logits, targets, num_classes=36)
                res_no_frcrn["condition"] = "No FRCRN"
                res_no_frcrn["model"] = model_name
                res_no_frcrn["params"] = p_count
                results_table.append(res_no_frcrn)
            except Exception as e:
                print(f"    [No FRCRN eval skipped]: {e}")

        # 2. Evaluate Mix + FRCRN (FRCRN Masked noise)
        if frcrn_cache_path and manifest_path:
            try:
                ds_frcrn = MaskedNoiseCacheDataset(
                    npy_path=frcrn_cache_path,
                    manifest_path=manifest_path,
                    apply_rms_norm=True,
                    target_rms=0.07925,
                    max_samples=args.max_samples,
                )
                loader_frcrn = torch.utils.data.DataLoader(
                    ds_frcrn, batch_size=args.batch_size, shuffle=False, num_workers=2
                )
                logits, targets, snrs = run_evaluation_for_model(model, loader_frcrn, device)
                res_frcrn = compute_metrics(logits, targets, num_classes=36)
                res_frcrn["condition"] = "Mix + FRCRN"
                res_frcrn["model"] = model_name
                res_frcrn["params"] = p_count
                results_table.append(res_frcrn)
            except Exception as e:
                print(f"    [Mix + FRCRN eval skipped]: {e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary table
    if results_table:
        print("\n" + "=" * 90)
        print("BENCHMARK COMPARISON TABLE (No FRCRN vs Mix + FRCRN)")
        print("=" * 90)
        print(f"{'Condition':<14} | {'Model':<20} | {'Params':<10} | {'Acc':<7} | {'Prec':<7} | {'Recall':<7} | {'Macro-F1':<8} | {'Micro-F1':<8} | {'mAP':<7}")
        print("-" * 90)
        for r in results_table:
            print(
                f"{r['condition']:<14} | {r['model']:<20} | {r['params']:<10} | "
                f"{r['acc']:<7.4f} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | "
                f"{r['macro_f1']:<8.4f} | {r['micro_f1']:<8.4f} | {r['mAP']:<7.4f}"
            )
        print("=" * 90)


if __name__ == "__main__":
    main()
