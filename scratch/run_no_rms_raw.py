import os, sys, time, yaml, json
from pathlib import Path
import torch

MOE_PROJECT_ROOT = Path("/marimo/moe")
if str(MOE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_PROJECT_ROOT))

# 1. Update rms_normalize in src/utils/files.py to bypass when input_rms <= 0
files_py = MOE_PROJECT_ROOT / "src/utils/files.py"
files_code = files_py.read_text(encoding="utf-8")
if "if target_rms is None or target_rms <= 0:" not in files_code:
    updated_files_code = files_code.replace(
        'def rms_normalize(waveform, target_rms: float):',
        'def rms_normalize(waveform, target_rms: float):\n    if target_rms is None or target_rms <= 0:\n        return waveform'
    )
    files_py.write_text(updated_files_code, encoding="utf-8")
    print("Updated rms_normalize to bypass normalization when target_rms <= 0")

from train_students import train_stage, save_json

# 2. Config for No-RMS run
cfg_path = MOE_PROJECT_ROOT / "config/kd_ced_tiny.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg_no_rms = dict(cfg)
cfg_no_rms["teacher"] = dict(cfg.get("teacher", {}))
cfg_no_rms["teacher"]["input_rms"] = 0.0  # Tat RMS normalization

no_rms_cfg_path = MOE_PROJECT_ROOT / "config/kd_ced_tiny_no_rms.yaml"
with open(no_rms_cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg_no_rms, f)

# 3. Output dir for No-RMS run
NO_RMS_OUTPUT_ROOT = Path("/marimo/output/moe-no-rms")
NO_RMS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

data_root = Path("/datasets/mix-dataset-v2")
cache_dir = Path("/datasets/cache-frcrn")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = int(cfg.get("train", {}).get("seed", 42))

LOW_EPOCHS = int(cfg["train"]["epochs"])
BATCH_SIZE = int(cfg["train"]["batch_size"])
EVAL_BATCH_SIZE = int(cfg["train"]["eval_batch_size"])
NUM_WORKERS = int(cfg["train"].get("num_workers", 0))
HIGH_EPOCHS = 10

teacher_fold0 = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
teacher_ema = Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")

print("=" * 70)
print("=== BAT DAU LUOT CHAY 2: KHONG CHUAN HOA RMS (NO RMS ABLATION) ===")
print(f"Device: {device}")
print(f"Data root: {data_root}")
print(f"Cache dir: {cache_dir}")
print(f"RMS target: 0.0 (BYPASSED / KHONG CHUAN HOA)")
print(f"Output root: {NO_RMS_OUTPUT_ROOT}")
print("=" * 70)

t0 = time.time()

# Stage 1: Low-Student without RMS
print("\n[LUOT 2 - GIAI DOAN 1/2] Huan luyen Low-Student KHONG RMS (15 epochs) voi tqdm:")
low_ckpt_no_rms, low_summary_no_rms = train_stage(
    name="Low-Student",
    snr_filter=None,
    teacher_checkpoint=teacher_fold0,
    teacher_config=MOE_PROJECT_ROOT / "config/beat.yaml",
    raw=cfg_no_rms,
    config_path=no_rms_cfg_path,
    cache_dir=cache_dir,
    output_dir=NO_RMS_OUTPUT_ROOT / "low",
    device=device,
    seed=seed,
    epochs=LOW_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# Stage 2: High-Student without RMS
print("\n[LUOT 2 - GIAI DOAN 2/2] Chuyen biet hoa High-Student KHONG RMS (SNR=20 dB, 10 epochs) voi tqdm:")
high_ckpt_no_rms, high_summary_no_rms = train_stage(
    name="High-Student",
    snr_filter=(20.0,),
    teacher_checkpoint=teacher_ema,
    teacher_config=MOE_PROJECT_ROOT / "config/beat.yaml",
    raw=cfg_no_rms,
    config_path=no_rms_cfg_path,
    cache_dir=cache_dir,
    output_dir=NO_RMS_OUTPUT_ROOT / "high",
    device=device,
    seed=seed + 1,
    epochs=HIGH_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    initialize_from=low_ckpt_no_rms,
)

save_json(NO_RMS_OUTPUT_ROOT / "summary.json", {"low": low_summary_no_rms, "high": high_summary_no_rms})
dur_no_rms = (time.time() - t0) / 60.0
print(f"\nLUOT CHAY 2 HOAN TAT trong {dur_no_rms:.1f} phut!")
print(f"Low checkpoint (No RMS): {low_ckpt_no_rms}")
print(f"High checkpoint (No RMS): {high_ckpt_no_rms}")
