"""
Benchmark Script for Task 3:
1. Trains BEATs -> CED-Tiny (nguyên bản) with 5 KD methods:
   - logit
   - crd
   - mgd
   - logit + crd
   - logit + mgd
2. Selects the best performing KD method based on validation Macro-F1 / mAP.
3. Evaluates and compares with CED-Tiny (có khối beat bias và post LN) using the best KD method.
4. Generates comprehensive comparison tables and saves logs & JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Safe stdout/stderr encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataset import get_dataloader, get_dataset
from src.kd import KDTrainer
from src.models import build_model
from src.utils import count_parameters, set_seed, setup_logger

KD_METHODS = ["logit", "crd", "mgd", "logit+crd", "logit+mgd"]


def load_teacher_model(device: torch.device, data_root: str | None = None) -> torch.nn.Module:
    """Loads BEATs teacher model with best available pretrained weights."""
    teacher = build_model("beats", num_classes=36).to(device)

    # Check candidates for teacher weights
    candidates = [
        PACKAGE_ROOT.parent.parent / "pretrained" / "beat-ft" / "beats_teacher_fold0_weights.pt",
        PACKAGE_ROOT / "pretrained" / "beats_teacher_fold0_weights.pt",
        PACKAGE_ROOT / "checkpoints" / "beats" / "best_model.pt",
    ]

    loaded = False
    for cand in candidates:
        if cand.is_file():
            try:
                ckpt = torch.load(cand, map_location=device, weights_only=False)
                if "state_dict" in ckpt:
                    # Filter keys if needed
                    sd = ckpt["state_dict"]
                    teacher.load_state_dict(sd, strict=False)
                elif "model_state_dict" in ckpt:
                    teacher.load_state_dict(ckpt["model_state_dict"])
                else:
                    teacher.load_state_dict(ckpt, strict=False)
                print(f"[+] Loaded Teacher weights from: {cand}")
                loaded = True
                break
            except Exception as e:
                print(f"[!] Could not load candidate {cand}: {e}")

    if not loaded:
        print("[*] No pre-trained checkpoint found; using initialized BEATs teacher baseline.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def run_single_kd_experiment(
    method: str,
    student_variant: str,
    teacher: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Runs a single Knowledge Distillation experiment."""
    variant_key = "ced_tiny" if student_variant == "vanilla" else "ced_tiny_beat_bias_postln"
    student = build_model(variant_key, num_classes=36).to(device)

    name = run_name or f"kd_{method.replace('+', '_')}_{student_variant}"
    logger.info("\n" + "=" * 86)
    logger.info("STARTING EXPERIMENT: Method='%s' | Student Variant='%s' (%s)",
                method.upper(), student_variant.upper(), variant_key)
    logger.info("=" * 86)

    trainer = KDTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        method=method,
        config=config,
        device=device,
        logger=logger,
        run_name=name,
    )
    result = trainer.fit()
    result["student_variant"] = student_variant
    result["model_key"] = variant_key
    total_p, _ = count_parameters(student)
    result["params_m"] = round(total_p / 1e6, 2)
    return result


def run_kd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = PACKAGE_ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI Overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.smoke:
        config.setdefault("training", {})["epochs"] = 1
        config.setdefault("training", {})["warmup_epochs"] = 0
        config.setdefault("training", {})["patience"] = 1

    device_str = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PACKAGE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"kd_benchmark_{timestamp}.log"
    logger = setup_logger(f"kd_benchmark_{timestamp}", log_file=log_file)

    logger.info("=" * 86)
    logger.info("TASK 3: BEATs -> CED-TINY KNOWLEDGE DISTILLATION BENCHMARK")
    logger.info("Timestamp: %s | Device: %s | Smoke: %s | Epochs: %d",
                timestamp, device, args.smoke, config.get("training", {}).get("epochs", 25))
    logger.info("5 Methods: logit, CRD, MGD, logit+CRD, logit+MGD")
    logger.info("=" * 86)

    # Build datasets
    data_root = args.data_root or config.get("dataset", {}).get("data_root")
    noise_cache = args.noise_cache or config.get("noise_cache", {}).get("cache_file") or config.get("noise_cache", {}).get("cache_dir")
    max_samples = 48 if args.smoke else None
    batch_size = config.get("training", {}).get("batch_size", 32) if not args.smoke else 16

    train_ds = get_dataset(
        root=data_root,
        split="train",
        max_samples=max_samples,
        noise_cache_path=noise_cache,
    )
    val_ds = get_dataset(
        root=data_root,
        split="validation",
        max_samples=max_samples,
        noise_cache_path=noise_cache,
    )

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info("Loaded Train Samples: %d | Val Samples: %d | Batch Size: %d", len(train_ds), len(val_ds), batch_size)

    # Load Teacher
    teacher = load_teacher_model(device, data_root=data_root)

    methods_to_run = KD_METHODS if args.method == "all" else [args.method]
    vanilla_results: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # GIAI ĐOẠN 1: Huấn luyện 5 phương pháp KD trên CED-tiny (nguyên mẫu)
    # -------------------------------------------------------------
    logger.info("\n" + "#" * 86)
    logger.info("GIAI DOAN 1: HUAN LUYEN 5 PHUONG PHAP KD TREN CED-TINY (NGUYEN MAU)")
    logger.info("#" * 86)

    for m_name in methods_to_run:
        res = run_single_kd_experiment(
            method=m_name,
            student_variant="vanilla",
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            logger=logger,
        )
        vanilla_results.append(res)

    # Bảng tổng hợp so sánh 5 phương pháp KD
    logger.info("\n" + "=" * 86)
    logger.info("BANG SO SANH 5 PHUONG PHAP KD TREN CED-TINY (NGUYEN MAU)")
    logger.info("=" * 86)
    logger.info("  | Method          | Params(M) | Accuracy | Precision |  Recall  | Macro-F1 | Micro-F1 |   mAP    |")
    logger.info("  |-----------------|-----------|----------|-----------|----------|----------|----------|----------|")
    for r in sorted(vanilla_results, key=lambda x: x["val_metrics"].get("macro_f1", 0.0), reverse=True):
        m = r["val_metrics"]
        logger.info(
            "  | %-15s | %9.2f | %7.2f%% | %9.4f | %8.4f | %8.4f | %8.4f | %8.4f |",
            r["method"],
            r["params_m"],
            m.get("acc", 0.0) * 100,
            m.get("precision", 0.0),
            m.get("recall", 0.0),
            m.get("macro_f1", 0.0),
            m.get("micro_f1", 0.0),
            m.get("mAP", 0.0),
        )
    logger.info("=" * 86)

    # -------------------------------------------------------------
    # GIAI ĐOẠN 2: Chọn phương pháp KD cao nhất
    # -------------------------------------------------------------
    best_vanilla = max(vanilla_results, key=lambda x: x["val_metrics"].get("macro_f1", 0.0))
    best_method = best_vanilla["method"]
    logger.info("\n" + "#" * 86)
    logger.info("GIAI DOAN 2: PHUONG PHAP KD DAT KET QUA CAO NHAT LA: '%s' (Macro-F1: %.4f, mAP: %.4f)",
                best_method.upper(), best_vanilla["val_metrics"].get("macro_f1", 0.0),
                best_vanilla["val_metrics"].get("mAP", 0.0))
    logger.info("#" * 86)

    # -------------------------------------------------------------
    # GIAI ĐOẠN 3: So sánh với CED-tiny có khối beat bias và post LN
    # -------------------------------------------------------------
    logger.info("\n" + "#" * 86)
    logger.info("GIAI DOAN 3: HUAN LUYEN CED-TINY (CO BEAT BIAS VA POST LN) VOI PHUONG PHAP '%s'", best_method.upper())
    logger.info("#" * 86)

    res_beat_bias = run_single_kd_experiment(
        method=best_method,
        student_variant="beat_bias_post_ln",
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        run_name=f"kd_{best_method.replace('+', '_')}_beat_bias_postln",
    )

    # -------------------------------------------------------------
    # GIAI ĐOẠN 4: BẢNG SO SÁNH CUỐI CÙNG & TÍNH ĐỘ LỆCH (DELTA)
    # -------------------------------------------------------------
    m_vanilla = best_vanilla["val_metrics"]
    m_bias = res_beat_bias["val_metrics"]

    logger.info("\n" + "=" * 96)
    logger.info("BANG SO SANH CHUYEN DOI: CED-TINY NGUYEN MAU VS CED-TINY (BEAT BIAS + POST LN)")
    logger.info("Phuong phap KD toi uu duoc ap dung: %s", best_method.upper())
    logger.info("=" * 96)
    logger.info("  | Architecture                  | Params(M) | Accuracy | Precision |  Recall  | Macro-F1 | Micro-F1 |   mAP    |")
    logger.info("  |-------------------------------|-----------|----------|-----------|----------|----------|----------|----------|")
    logger.info(
        "  | %-29s | %9.2f | %7.2f%% | %9.4f | %8.4f | %8.4f | %8.4f | %8.4f |",
        "CED-Tiny (Nguyen Mau)",
        best_vanilla["params_m"],
        m_vanilla.get("acc", 0.0) * 100,
        m_vanilla.get("precision", 0.0),
        m_vanilla.get("recall", 0.0),
        m_vanilla.get("macro_f1", 0.0),
        m_vanilla.get("micro_f1", 0.0),
        m_vanilla.get("mAP", 0.0),
    )
    logger.info(
        "  | %-29s | %9.2f | %7.2f%% | %9.4f | %8.4f | %8.4f | %8.4f | %8.4f |",
        "CED-Tiny (BeatBias + PostLN)   ",
        res_beat_bias["params_m"],
        m_bias.get("acc", 0.0) * 100,
        m_bias.get("precision", 0.0),
        m_bias.get("recall", 0.0),
        m_bias.get("macro_f1", 0.0),
        m_bias.get("micro_f1", 0.0),
        m_bias.get("mAP", 0.0),
    )

    d_acc = (m_bias.get("acc", 0.0) - m_vanilla.get("acc", 0.0)) * 100
    d_f1 = m_bias.get("macro_f1", 0.0) - m_vanilla.get("macro_f1", 0.0)
    d_map = m_bias.get("mAP", 0.0) - m_vanilla.get("mAP", 0.0)
    logger.info("  | %-29s | %9s | %+6.2f%% | %9s | %8s | %+8.4f | %8s | %+8.4f |",
                "Delta (Improvement)", "-", d_acc, "-", "-", d_f1, "-", d_map)
    logger.info("=" * 96)

    # Detailed SNR comparison
    logger.info("\n" + "=" * 96)
    logger.info("CHI TIET PHAN RA THEO TUNG MUC SNR (MACRO-F1 & mAP):")
    logger.info("  | SNR Level | Vanilla F1 | Bias+PostLN F1 | Delta F1 | Vanilla mAP | Bias+PostLN mAP | Delta mAP |")
    logger.info("  |-----------|------------|----------------|----------|-------------|-----------------|-----------|")
    for snr in ["-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"]:
        v_snr = best_vanilla.get("snr_breakdown", {}).get(snr, {})
        b_snr = res_beat_bias.get("snr_breakdown", {}).get(snr, {})
        v_f1 = v_snr.get("macro_f1", 0.0)
        b_f1 = b_snr.get("macro_f1", 0.0)
        v_map = v_snr.get("mAP", 0.0)
        b_map = b_snr.get("mAP", 0.0)
        logger.info(
            "  | %9s | %10.4f | %14.4f | %+8.4f | %11.4f | %15.4f | %+9.4f |",
            snr, v_f1, b_f1, b_f1 - v_f1, v_map, b_map, b_map - v_map
        )
    logger.info("=" * 96)

    # Save summary
    summary_data = {
        "timestamp": timestamp,
        "device": str(device),
        "smoke": args.smoke,
        "best_method": best_method,
        "vanilla_benchmark": vanilla_results,
        "beat_bias_postln_result": res_beat_bias,
        "comparison_delta": {
            "delta_acc": round(d_acc, 4),
            "delta_macro_f1": round(d_f1, 4),
            "delta_map": round(d_map, 4),
        },
    }
    summary_path = log_dir / "kd_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("\nSaved full KD benchmark summary to: %s", summary_path)
    logger.info("Full log file saved to: %s", log_file)
    print(f"\n[OK] Task 3 benchmark completed successfully! Best KD Method: {best_method}")
    print(f"     Full Log: {log_file}")
    print(f"     Summary:  {summary_path}")
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Review-3 Task 3: 5 KD Methods Benchmark on CED-Tiny")
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "logit", "crd", "mgd", "logit+crd", "logit+mgd"],
                        help="Specific KD method or 'all' to run complete benchmark")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--smoke", action="store_true", help="Quick 1-epoch smoke test with small sample count")
    parser.add_argument("--noise-cache", type=str, default=None, help="Path to FRCRN-masked .npy cache or directory")
    parser.add_argument("--data-root", type=str, default=None, help="Path to mixture dataset root")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_kd_benchmark(args)


if __name__ == "__main__":
    main()
