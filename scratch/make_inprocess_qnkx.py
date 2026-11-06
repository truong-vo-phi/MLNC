import base64

cell17_inprocess = """import sys, os, time, yaml
from pathlib import Path
import torch

if not MOE_TRAIN_READY:
    raise RuntimeError("MoE training prerequisites are missing; see the preflight cell.")

# Ensure imp/moe is in python sys.path
_moe_dir = str(MOE_PROJECT_ROOT)
if _moe_dir not in sys.path:
    sys.path.insert(0, _moe_dir)

from train_students import train_stage, update_moe_checkpoint_config, save_json
from scripts.load_dataset import resolve_dataset_root

# Load its own config strictly from config/kd_ced_tiny.yaml
_cfg_path = MOE_PROJECT_ROOT / "config/kd_ced_tiny.yaml"
with open(_cfg_path, encoding="utf-8") as _f:
    moe_config = yaml.safe_load(_f)

# Resolve data and cache paths from config
_data_root = Path(MOE_DATA_ROOT)
_cache_dir = Path(MOE_CACHE_DIR)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seed = int(moe_config.get("train", {}).get("seed", 42))

# Training parameters directly from config/kd_ced_tiny.yaml (Khong override tu mix_dataset_v2)
LOW_EPOCHS = int(moe_config["train"]["epochs"])
BATCH_SIZE = int(moe_config["train"]["batch_size"])
EVAL_BATCH_SIZE = int(moe_config["train"]["eval_batch_size"])
NUM_WORKERS = int(moe_config["train"].get("num_workers", 0))
HIGH_EPOCHS = 10  # Chuyen gia High-SNR fine-tuning

print(f"=== HUAN LUYEN THEO CONFIG CUA CHINH NO ({_cfg_path.name}) ===")
print(f"Device: {_device}")
print(f"Data root: {_data_root}")
print(f"Cache dir: {_cache_dir}")
print(f"Low Student epochs: {LOW_EPOCHS} | High Student epochs: {HIGH_EPOCHS}")
print(f"Batch size: {BATCH_SIZE} | Eval batch size: {EVAL_BATCH_SIZE}")
print(f"Learning rate: {moe_config['train']['lr']} | Weight decay: {moe_config['train']['weight_decay']}")
print(f"Distillation: T={moe_config['loss']['temperature']}, weight={moe_config['loss']['weight']}, smoothing={moe_config['loss']['label_smoothing']}")
print(f"Selection metric: {moe_config['train']['selection_metric']}")
print("=" * 65)

mo.status.toast(f"🚀 Khoi dong huan luyen MoE theo config kd_ced_tiny.yaml voi tqdm...", kind="info")
_t0 = time.time()

# --------------------------------------------------------------------------
# Stage 1: Low-Student (toan bo dai SNR, Teacher Fold0) theo kd_ced_tiny.yaml
# --------------------------------------------------------------------------
print("\\n[GIAI DOAN 1/2] Huấn luyện Low-Student (15 epochs) với tqdm:")
low_ckpt, low_summary = train_stage(
    name="Low-Student",
    snr_filter=None,
    teacher_checkpoint=MOE_TEACHER_FOLD0,
    teacher_config=MOE_PROJECT_ROOT / "config/beat.yaml",
    raw=moe_config,
    config_path=_cfg_path,
    cache_dir=_cache_dir,
    output_dir=MOE_OUTPUT_ROOT / "low",
    device=_device,
    seed=_seed,
    epochs=LOW_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# --------------------------------------------------------------------------
# Stage 2: High-Student (chuyen biet hoa SNR=20 dB, Teacher EMA)
# --------------------------------------------------------------------------
print("\\n[GIAI DOAN 2/2] Chuyên biệt hóa High-Student (SNR=20 dB, 10 epochs) với tqdm:")
high_ckpt, high_summary = train_stage(
    name="High-Student",
    snr_filter=(20.0,),
    teacher_checkpoint=MOE_TEACHER_EMA,
    teacher_config=MOE_PROJECT_ROOT / "config/beat.yaml",
    raw=moe_config,
    config_path=_cfg_path,
    cache_dir=_cache_dir,
    output_dir=MOE_OUTPUT_ROOT / "high",
    device=_device,
    seed=_seed + 1,
    epochs=HIGH_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    initialize_from=low_ckpt,
)

# Cap nhat config/moe.yaml va luu summary
update_moe_checkpoint_config(MOE_PROJECT_ROOT / "config/moe.yaml", low_ckpt, high_ckpt)
save_json(MOE_OUTPUT_ROOT / "summary.json", {"low": low_summary, "high": high_summary})

MOE_TRAINING_COMPLETE = True
_dur_min = (time.time() - _t0) / 60.0

mo.status.toast(f"🎉 Huan luyen hoan tat trong {_dur_min:.1f} phut!", kind="success")
print(f"\\nTRAINING COMPLETE! Output root: {MOE_OUTPUT_ROOT}")
print(f"Low checkpoint: {low_ckpt}")
print(f"High checkpoint: {high_ckpt}")

mo.md(f\"\"\"
### 🎉 Huấn Luyện Hai Giai Đoạn Hoàn Tất (Theo Config Của Chính Nó)

> 🟢 **Trạng thái:** Hoàn tất thành công 100% bằng **tqdm** trực tiếp trong kernel.  
> ⏱️ **Thời gian:** `{_dur_min:.1f} phút` trên thiết bị `{_device}`.  
> 📁 **Thư mục đầu ra:** `{MOE_OUTPUT_ROOT}`

| Hạng mục | Đường dẫn tệp tin |
| :--- | :--- |
| **Cấu hình huấn luyện** | `{_cfg_path}` |
| **Low Student Checkpoint** | `{low_ckpt}` |
| **High Student Checkpoint** | `{high_ckpt}` |
| **Tổng kết huấn luyện (Summary)** | `{MOE_OUTPUT_ROOT}/summary.json` |
\"\"\")
"""

b64_c17 = base64.b64encode(cell17_inprocess.encode("utf-8")).decode("ascii")

update_script = f"""import base64
import marimo._code_mode as cm

code = base64.b64decode("{b64_c17}").decode("utf-8")
async with cm.get_context() as ctx:
    ctx.edit_cell("qnkX", code=code)

print("SUCCESS_EDIT_IN_PROCESS_QNKX")
"""

with open("d:/PROJECT/MLNC/scratch/deploy_inprocess_qnkx.py", "w", encoding="ascii") as f:
    f.write(update_script)
print("Wrote deploy_inprocess_qnkx.py")
