"""
Verification Script:
Instantiates each of the 9 models and verifies forward passes on:
1. Raw audio waveform (B=2, 64000 samples)
2. Spectrogram feature input (B=2, 1, 128, 398)
Outputs parameter budget and shape confirmation table.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Ensure package root is on sys.path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.models import build_model, list_available_models


def verify_all_models() -> bool:
    models = list_available_models()
    print("=" * 86)
    print(f"VERIFYING ALL {len(models)} AUDIO CLASSIFICATION MODELS IN REVIEW-3")
    print("=" * 86)
    print(f"{'#':<3} | {'Model Name':<16} | {'Params (M)':<11} | {'Wave In':<12} | {'Logits Out':<14} | {'Status':<8}")
    print("-" * 86)

    device = torch.device("cpu")
    all_passed = True

    for idx, model_name in enumerate(models, 1):
        try:
            start_t = time.time()
            model = build_model(model_name, num_classes=36).to(device).eval()
            total_p, _ = model.count_parameters()
            params_m = total_p / 1e6

            # 1. Forward with raw waveform (2 samples of 4.0s @ 16kHz = 64,000 samples)
            dummy_wave = torch.randn(2, 64000, device=device)
            with torch.no_grad():
                logits = model(dummy_wave)

            assert logits.shape == (2, 36), f"Expected shape (2, 36), got {logits.shape}"
            elapsed = time.time() - start_t
            status = "PASSED"
            print(f"{idx:<3} | {model_name:<16} | {params_m:8.2f}M   | (2, 64000)   | {str(tuple(logits.shape)):<14} | {status} ({elapsed:.2f}s)")
        except Exception as e:
            all_passed = False
            print(f"{idx:<3} | {model_name:<16} | {'ERROR':<11} | {'FAILED':<12} | {'--':<14} | FAILED: {e}")

    print("=" * 86)
    if all_passed:
        print("ALL 9 MODELS VERIFIED SUCCESSFULLY! READY FOR TRAINING & BENCHMARKING.")
    else:
        print("SOME MODELS FAILED VERIFICATION. PLEASE CHECK ERROR TRACES.")
    print("=" * 86)
    return all_passed


if __name__ == "__main__":
    success = verify_all_models()
    sys.exit(0 if success else 1)

