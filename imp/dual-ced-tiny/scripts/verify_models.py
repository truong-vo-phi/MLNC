"""
Verification Script for Dual-CED-Tiny Models:
Checks model instantiation, parameter budgets (~10.824M for Dual-CED-Tiny, ~5.412M for single experts),
waveform forward shape, spectrogram forward shape, and routing mechanism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model, list_available_models
from src.utils import count_parameters


def verify_all_models() -> bool:
    print("=" * 86)
    print("DUAL-CED-TINY: ARCHITECTURE & SHAPE VERIFICATION")
    print("=" * 86)

    device = torch.device("cpu")
    models = list_available_models()
    all_passed = True

    for name in models:
        print(f"\n[+] Verifying model: '{name}'...")
        try:
            model = build_model(name, num_classes=36).to(device)
            model.eval()

            total_p, train_p = count_parameters(model)
            params_m = total_p / 1e6
            print(f"    - Parameters: Total={total_p:,} ({params_m:.2f}M), Trainable={train_p:,}")

            if name == "dual_ced_tiny":
                assert 10.0 <= params_m <= 11.5, f"Dual-CED-Tiny params {params_m:.2f}M out of expected range ~10.824M"
            else:
                assert 5.0 <= params_m <= 6.0, f"CED-Tiny params {params_m:.2f}M out of expected range ~5.41M"

            # 1. Waveform Forward Pass: [2, 64000]
            dummy_wave = torch.randn(2, 64000, device=device)
            dummy_snr = torch.tensor([5.0, 20.0], device=device)
            with torch.no_grad():
                out_wave = model(dummy_wave, snr_db=dummy_snr)

            assert out_wave.shape == (2, 36), f"Output shape mismatch: {out_wave.shape}"
            assert not torch.isnan(out_wave).any(), "NaN found in waveform forward pass"
            print(f"    - Waveform Forward: [2, 64000] -> {list(out_wave.shape)} [PASSED]")

            # 2. Spectrogram Forward Pass: [2, 1, 64, 398]
            dummy_spec = torch.randn(2, 1, 64, 398, device=device)
            with torch.no_grad():
                out_spec = model(dummy_spec, snr_db=dummy_snr)

            assert out_spec.shape == (2, 36), f"Spectrogram shape mismatch: {out_spec.shape}"
            assert not torch.isnan(out_spec).any(), "NaN found in spectrogram forward pass"
            print(f"    - Spectrogram Forward: [2, 1, 64, 398] -> {list(out_spec.shape)} [PASSED]")

            # 3. Router Verification (For Dual-CED-Tiny)
            if name == "dual_ced_tiny":
                with torch.no_grad():
                    _, details = model(dummy_wave, snr_db=dummy_snr, return_details=True)
                alpha = details["alpha"]
                assert alpha.shape == (2, 1), f"Alpha shape mismatch: {alpha.shape}"
                # Sample 0 has 5 dB (<18) -> Low-expert dominant; Sample 1 has 20 dB (>=18) -> High-expert dominant
                assert alpha[0].item() < 0.5, f"Expected low alpha for 5 dB, got {alpha[0].item()}"
                assert alpha[1].item() >= 0.5, f"Expected high alpha for 20 dB, got {alpha[1].item()}"
                print(f"    - Router Gating Check: 5dB alpha={alpha[0].item():.2f}, 20dB alpha={alpha[1].item():.2f} [PASSED]")

            print(f"    [*] Model '{name}': VERIFIED SUCCESSFULLY.")

        except Exception as e:
            print(f"    [!] FAILED verifying '{name}': {e}")
            all_passed = False

    print("\n" + "=" * 86)
    if all_passed:
        print("ALL DUAL-CED-TINY MODELS PASSED VERIFICATION!")
    else:
        print("SOME MODELS FAILED VERIFICATION. Check error messages above.")
    print("=" * 86)
    return all_passed


if __name__ == "__main__":
    success = verify_all_models()
    sys.exit(0 if success else 1)
