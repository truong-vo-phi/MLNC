import base64
import marimo._code_mode as cm

cell17_code = """import os as _os, subprocess as _subprocess, sys as _sys, re as _re, time as _time
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
_stage = "Stage 1/2: Low-Student"
_t_start = _time.time()

mo.status.toast(f"🚀 Khởi động huấn luyện Two-Stage MoE ({_total_epochs} epochs)...", kind="info")
print("STARTING FRESH TWO-STAGE TRAINING:", " ".join(MOE_TRAIN_COMMAND), flush=True)

with mo.status.progress_bar(total=_total_epochs, title="Huấn Luyện Dual-Expert CED-Tiny MoE") as _pbar:
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

            # Tracking stage & epoch
            if "[Low-Student] START" in _line_str:
                _stage = "Stage 1: Low-Student (15 epochs)"
                _pbar.update(subtitle=_stage)
            elif "[High-Student] START" in _line_str:
                _stage = "Stage 2: High-Student (10 epochs)"
                _pbar.update(subtitle=_stage)
            elif "] epoch=" in _line_str and ("loss=" in _line_str or "val_" in _line_str):
                _match = _re.search(r'epoch=(\\d+)', _line_str)
                _ep_str = f"Epoch {_match.group(1)}" if _match else ""
                _pbar.update(increment=1, subtitle=f"{_stage} | {_ep_str}")
            elif "COMPLETE best_epoch=" in _line_str:
                _pbar.update(subtitle=f"✓ Hoàn tất {_stage}")

        _code = _proc.wait()

if _code:
    mo.status.toast(f"❌ Huấn luyện thất bại với mã lỗi exit={_code}", kind="danger")
    raise RuntimeError(f"Training failed with exit={_code}; see {MOE_TRAIN_LOG}")

MOE_TRAINING_COMPLETE = True
_elapsed_min = (_time.time() - _t_start) / 60.0
mo.status.toast(f"🎉 Huấn luyện thành công trong {_elapsed_min:.1f} phút!", kind="success")

MOE_TRAIN_VIEW = mo.md(f\"\"\"
### 🎉 Huấn Luyện Hai Giai Đoạn Hoàn Tất!

> 🟢 **Trạng thái:** Huấn luyện thành công cả **Low-Student** và **High-Student** trên thiết bị CUDA.  
> ⏱️ **Tổng thời gian:** `{_elapsed_min:.1f} phút` | **Thư mục đầu ra:** `{MOE_OUTPUT_ROOT}`

| Hạng mục | Đường dẫn Checkpoint / Tệp tin |
| :--- | :--- |
| **Low Student Checkpoint** | `{MOE_OUTPUT_ROOT}/low/last_best_checkpoint.pt` |
| **High Student Checkpoint** | `{MOE_OUTPUT_ROOT}/high/last_best_checkpoint.pt` |
| **Nhật ký huấn luyện (Training Log)** | `{MOE_TRAIN_LOG}` |
\"\"\")
MOE_TRAIN_VIEW
"""

async with cm.get_context() as ctx:
    ctx.edit_cell("qnkX", code=cell17_code)

print("SUCCESSFULLY_UPDATED_CELL_17")
