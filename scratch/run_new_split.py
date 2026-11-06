import os, sys, time, yaml, json
from pathlib import Path
import torch

MOE_PROJECT_ROOT = Path("/marimo/moe")
if str(MOE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_PROJECT_ROOT))

from train_students import train_stage, save_json

# 1. Config with RMS normalization (winner setting: input_rms = 0.07925)
cfg_path = MOE_PROJECT_ROOT / "config/kd_ced_tiny.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg_split = dict(cfg)
cfg_split["teacher"] = dict(cfg.get("teacher", {}))
cfg_split["teacher"]["input_rms"] = 0.07925  # Chuan hoa RMS chuan

split_cfg_path = MOE_PROJECT_ROOT / "config/kd_ced_tiny_split_10_15.yaml"
with open(split_cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg_split, f)

# 2. Output dir for New Split Run
SPLIT_OUTPUT_ROOT = Path("/marimo/output/moe-split-10-15")
SPLIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

data_root = Path("/datasets/mix-dataset-v2")
cache_dir = Path("/datasets/cache-frcrn")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = int(cfg.get("train", {}).get("seed", 42))

LOW_EPOCHS = int(cfg["train"]["epochs"]) # 15
BATCH_SIZE = int(cfg["train"]["batch_size"]) # 32
EVAL_BATCH_SIZE = int(cfg["train"]["eval_batch_size"]) # 64
NUM_WORKERS = int(cfg["train"].get("num_workers", 0))
HIGH_EPOCHS = 10

teacher_fold0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
teacher_ema = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")

print("=" * 80)
print("=== BAT DAU LUOT CHAY 3: CHIA EXPERT MOI (LOW: -5,0,5,10 dB | HIGH: 15,20 dB) ===")
print(f"Device: {device}")
print(f"Data root: {data_root}")
print(f"Cache dir: {cache_dir}")
print(f"RMS target: 0.07925 (Co chuan hoa RMS - Winner)")
print(f"Low SNRs: (-5.0, 0.0, 5.0, 10.0) | High SNRs: (15.0, 20.0)")
print(f"Output root: {SPLIT_OUTPUT_ROOT}")
print("=" * 80)

t0 = time.time()

# Stage 1: Low-Student on SNRs [-5, 0, 5, 10]
print("\n[LUOT 3 - GIAI DOAN 1/2] Huan luyen Low-Student (SNR: -5, 0, 5, 10 dB) (15 epochs):")
low_ckpt, low_summary = train_stage(
    name="Low-Student",
    snr_filter=(-5.0, 0.0, 5.0, 10.0),
    teacher_checkpoint=teacher_fold0,
    teacher_config=MOE_PROJECT_ROOT / "config/beat.yaml",
    raw=cfg_split,
    config_path=split_cfg_path,
    cache_dir=cache_dir,
    output_dir=SPLIT_OUTPUT_ROOT / "low",
    device=device,
    seed=seed,
    epochs=LOW_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# Stage 2: High-Student on SNRs [15, 20]
print("\n[LUOT 3 - GIAI DOAN 2/2] Chuyen biet hoa High-Student (SNR: 15, 20 dB) (10 epochs):")
high_ckpt, high_summary = train_stage(
    name="High-Student",
    snr_filter=(15.0, 20.0),
    teacher_checkpoint=teacher_ema,
    teacher_config=MOE_PROJECT_ROOT / "config/beat.yaml",
    raw=cfg_split,
    config_path=split_cfg_path,
    cache_dir=cache_dir,
    output_dir=SPLIT_OUTPUT_ROOT / "high",
    device=device,
    seed=seed + 1,
    epochs=HIGH_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    initialize_from=low_ckpt,
)

save_json(SPLIT_OUTPUT_ROOT / "summary.json", {"low": low_summary, "high": high_summary})
dur = (time.time() - t0) / 60.0
print(f"\nLUOT CHAY 3 HOAN TAT trong {dur:.1f} phut!")
print(f"Low checkpoint: {low_ckpt}")
print(f"High checkpoint: {high_ckpt}")
