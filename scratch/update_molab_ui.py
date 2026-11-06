import marimo._code_mode as cm
import base64

# Code for Cell 15 (Preflight & Environment Check with Rich Visual Status)
cell15_code = '''import csv as _csv
import numpy as _np
from pathlib import Path as _Path

try:
    import torch as _torch
    import torchaudio as _torchaudio
    from torchaudio._extension.utils import _check_cuda_version as _check_ta_cuda_version
    _check_ta_cuda_version()
    MOE_AUDIO_RUNTIME_OK = bool(_torch.cuda.is_available())
    MOE_AUDIO_RUNTIME_INFO = f"torch={_torch.__version__} cuda={_torch.version.cuda} torchaudio={_torchaudio.__version__} gpu={_torch.cuda.get_device_name(0) if _torch.cuda.is_available() else 'unavailable'}"
except Exception as _audio_error:
    MOE_AUDIO_RUNTIME_OK = False
    MOE_AUDIO_RUNTIME_INFO = repr(_audio_error)

# Candidate paths for data root and cache (supporting both /datasets and /dataset)
for _cand in [Path("/datasets/mix-dataset-v2"), Path("/dataset/mix-dataset-v2"), Path("/marimo/dataset/mix-dataset-v2")]:
    if (_cand / "eval/manifest.csv").is_file():
        MOE_DATA_ROOT = _cand
        break
else:
    MOE_DATA_ROOT = Path("/datasets/mix-dataset-v2")

for _cand in [Path("/datasets/cache-frcrn"), Path("/dataset/cache"), Path("/datasets/cache"), Path("/marimo/dataset/cache")]:
    if (_cand / "train.npy").is_file():
        MOE_CACHE_DIR = _cand
        break
else:
    MOE_CACHE_DIR = Path("/datasets/cache-frcrn")

MOE_LOW_CHECKPOINT = MOE_OUTPUT_ROOT / "low/last_best_checkpoint.pt"
MOE_HIGH_CHECKPOINT = MOE_OUTPUT_ROOT / "high/last_best_checkpoint.pt"
MOE_TEACHER_FOLD0 = _Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
MOE_TEACHER_EMA = _Path("/marimo/pretrained/beats_teacher_ema_standalone.pt")

MOE_REQUIRED = {
    "Low student checkpoint": MOE_LOW_CHECKPOINT,
    "High student checkpoint": MOE_HIGH_CHECKPOINT,
    "BEATs Fold0 Teacher": MOE_TEACHER_FOLD0,
    "BEATs EMA Teacher": MOE_TEACHER_EMA,
    "Train manifest": MOE_DATA_ROOT / "train/manifest.csv",
    "Validation manifest": MOE_DATA_ROOT / "eval/manifest.csv",
    "FRCRN train cache": MOE_CACHE_DIR / "train.npy",
    "FRCRN validation cache": MOE_CACHE_DIR / "validation.npy",
}

MOE_PREFLIGHT = {name: path.is_file() for name, path in MOE_REQUIRED.items()}
MOE_PREFLIGHT["Torch/TorchAudio CUDA runtime"] = MOE_AUDIO_RUNTIME_OK
MOE_TRAIN_REQUIRED = {name: MOE_PREFLIGHT[name] for name in [
    "BEATs Fold0 Teacher", "BEATs EMA Teacher", "Train manifest",
    "Validation manifest", "FRCRN train cache", "FRCRN validation cache",
    "Torch/TorchAudio CUDA runtime"
]}

if all(MOE_TRAIN_REQUIRED.values()):
    def _manifest_count(path):
        with path.open(encoding="utf-8-sig", newline="") as _h:
            return sum(1 for _ in _csv.DictReader(_h))
    MOE_MANIFEST_COUNTS = {
        "train": _manifest_count(MOE_DATA_ROOT / "train/manifest.csv"),
        "validation": _manifest_count(MOE_DATA_ROOT / "eval/manifest.csv")
    }
    MOE_CACHE_COUNTS = {
        "train": int(_np.load(MOE_CACHE_DIR / "train.npy", mmap_mode="r").shape[0]),
        "validation": int(_np.load(MOE_CACHE_DIR / "validation.npy", mmap_mode="r").shape[0])
    }
    MOE_DATA_ALIGNMENT_OK = MOE_MANIFEST_COUNTS == MOE_CACHE_COUNTS
else:
    MOE_MANIFEST_COUNTS = {}
    MOE_CACHE_COUNTS = {}
    MOE_DATA_ALIGNMENT_OK = False

MOE_TRAIN_READY = all(MOE_TRAIN_REQUIRED.values()) and MOE_DATA_ALIGNMENT_OK
MOE_READY = all(MOE_PREFLIGHT.values())

# Format Markdown status table
_rows = []
for _name, _ok in MOE_PREFLIGHT.items():
    _badge = "🟢 SẴN SÀNG" if _ok else "⏳ CHƯA CÓ"
    _path_str = str(MOE_REQUIRED.get(_name, MOE_AUDIO_RUNTIME_INFO))
    _rows.append(f"| **{_name}** | {_badge} | `{_path_str}` |")

_status_title = "🟢 **HỆ THỐNG SẴN SÀNG HUẤN LUYỆN DUAL-STUDENT MoE**" if MOE_TRAIN_READY else "⚠️ **ĐANG CHỜ TẢI ĐẦY ĐỦ TIỀN ĐIỀU KIỆN**"

mo.md(f"""
### 📋 Kiểm Tra Điều Kiện Môi Trường & Dữ Liệu (Preflight Check)

> {_status_title}  
> 🖥️ **GPU Runtime:** `{MOE_AUDIO_RUNTIME_INFO}`  
> 📊 **Đồng bộ Dữ liệu:** Train = `{MOE_MANIFEST_COUNTS.get('train', 0):,}` clips | Eval = `{MOE_MANIFEST_COUNTS.get('validation', 0):,}` clips | Khớp cache: `{MOE_DATA_ALIGNMENT_OK}`

| Thành Phần Yêu Cầu | Trạng Thái | Đường Dẫn / Chi Tiết |
| :--- | :---: | :--- |
{chr(10).join(_rows)}
""")
'''

# Code for Cell 17 (Training execution with live progress bar and clean logging)
cell17_code = '''import os as _os, subprocess as _subprocess, sys as _sys, re as _re
if not MOE_TRAIN_READY:
    raise RuntimeError("MoE training prerequisites are missing; see the preflight cell.")

MOE_TRAIN_COMMAND = [
    _sys.executable, "-u", str(MOE_TRAIN_RUNNER),
    "--config", str(MOE_PROJECT_ROOT / "config/kd_ced_tiny.yaml"),
    "--data-root", str(MOE_DATA_ROOT),
    "--cache-dir", str(MOE_CACHE_DIR),
    "--fold0-checkpoint", str(MOE_TEACHER_FOLD0),
    "--ema-checkpoint", str(MOE_TEACHER_EMA),
    "--output-root", str(MOE_OUTPUT_ROOT),
    "--low-epochs", str(MOE_TRAIN_SETTINGS["low_epochs"]),
    "--high-epochs", str(MOE_TRAIN_SETTINGS["high_epochs"]),
    "--batch-size", str(MOE_TRAIN_SETTINGS["batch_size"]),
    "--eval-batch-size", str(MOE_TRAIN_SETTINGS["eval_batch_size"]),
    "--num-workers", str(MOE_TRAIN_SETTINGS["num_workers"]),
    "--seed", str(MOE_TRAIN_SETTINGS["seed"]),
    "--device", "cuda"
]

MOE_TRAIN_LOG = MOE_OUTPUT_ROOT / "training.log"
MOE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_env = _os.environ.copy()
_env["PYTHONUNBUFFERED"] = "1"
_env["MLNC_MIXTURE_36L"] = str(MOE_DATA_ROOT)

_total_epochs = MOE_TRAIN_SETTINGS["low_epochs"] + MOE_TRAIN_SETTINGS["high_epochs"]
_current_stage = "Khởi tạo"
_recent_logs = []

print("🚀 STARTING FRESH TWO-STAGE TRAINING:", " ".join(MOE_TRAIN_COMMAND), flush=True)

with mo.status.progress_bar(total=_total_epochs, title="⏳ Huấn Luyện Dual-Expert CED-Tiny MoE") as _pbar:
    with MOE_TRAIN_LOG.open("w", encoding="utf-8") as _log:
        _proc = _subprocess.Popen(
            MOE_TRAIN_COMMAND,
            cwd=str(MOE_PROJECT_ROOT),
            env=_env,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for _line in _proc.stdout:
            _line_str = _line.rstrip()
            print(_line_str, flush=True)
            _log.write(_line)
            _log.flush()
            _recent_logs.append(_line_str)
            if len(_recent_logs) > 10:
                _recent_logs.pop(0)

            # Phân tích tiến trình từ log
            if "[Low-Student] START" in _line_str:
                _current_stage = "Stage 1: Low-Student (15 epochs)"
                _pbar.update(subtitle=_current_stage)
            elif "[High-Student] START" in _line_str:
                _current_stage = "Stage 2: High-Student (10 epochs)"
                _pbar.update(subtitle=_current_stage)
            elif "epoch=" in _line_str and "loss=" in _line_str:
                _pbar.update(increment=1, subtitle=f"{_current_stage} | {_line_str[-60:]}")
            elif "COMPLETE best_epoch=" in _line_str:
                _pbar.update(subtitle=f"Hoàn thành {_current_stage}")

        _code = _proc.wait()

if _code:
    raise RuntimeError(f"Training failed with exit={_code}; see {MOE_TRAIN_LOG}")

MOE_TRAINING_COMPLETE = True
mo.status.toast("🎉 Huấn luyện Dual-Expert MoE hoàn tất thành công!", kind="success")

mo.md(f"""
### 🎉 Huấn Luyện Hai Giai Đoạn Hoàn Tất!

- **Giai đoạn 1:** Low-Student (`{MOE_TRAIN_SETTINGS['low_epochs']}` epochs) -> `{MOE_OUTPUT_ROOT}/low/last_best_checkpoint.pt`
- **Giai đoạn 2:** High-Student (`{MOE_TRAIN_SETTINGS['high_epochs']}` epochs) -> `{MOE_OUTPUT_ROOT}/high/last_best_checkpoint.pt`
- **Log chi tiết:** `{MOE_TRAIN_LOG}`
""")
'''

print("Generated code ready for injection.")
