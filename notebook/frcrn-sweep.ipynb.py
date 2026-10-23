import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell
def _():
    import os
    import sys
    from importlib.metadata import version as _installed_version
    from pathlib import Path

    import marimo as mo
    import torch

    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Cài gói (ví dụ timm) có thể nâng torch trên đĩa trong khi kernel vẫn giữ bản cũ trong bộ nhớ
    TORCH_MISMATCH = _installed_version("torch").split("+")[0] != torch.__version__.split("+")[0]
    return Path, TORCH_MISMATCH, mo, os, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Fine-tune FRCRN (target nhiễu) trên Mixture-36l: tối ưu tham số huấn luyện cho SNR cao (15-20 dB)

    Khởi tạo từ **FRCRN_SE_16K gốc** (`pretrained/last_best_checkpoint.pt`), `target: noise` (mạng học thẳng nhiễu, so với `speech` làm đối chứng), kiến trúc giữ nguyên.
    Chỉ đổi tham số huấn luyện (lr, decay, batch, segment, grad_clip, trọng số mask, lịch giảm lr).

    | Bước | Việc | Cờ |
    | --- | --- | --- |
    | 0 | Mốc: SI-SNR nhiễu và tiếng nói theo từng SNR của checkpoint gốc và `pretrained/FRCRN/best.pt` | luôn chạy (chỉ đánh giá) |
    | 1 | Đo thời gian: 1 epoch smoke để ước lượng thời gian mỗi epoch | `RUN_TIMING` |
    | 2 | Sweep W&B (random, Hyperband), chọn theo SI-SNR **nhiễu** trung bình 15 và 20 dB | `RUN_SWEEP` |
    | 3 | Xác nhận cấu hình tốt nhất: đủ epoch, nhiều seed, SI-SNR từng SNR, PESQ/STOI | `RUN_CONFIRM` |

    **Mặc định mọi cờ chạy nặng là `False`**: *Run all* chỉ dựng cấu hình, nạp dataset và đánh giá mốc.
    Bước huấn luyện và sweep chỉ chạy khi bạn tự đặt cờ thành `True`. Token W&B đọc từ biến môi trường
    `WANDB_API_KEY`, không ghi vào file.

    Cấu hình sweep: `config/frcrn_sweep.yaml`. Mã: `scripts/sweep_frcrn.py`, `scripts/finetune.py`.
    """)
    return


@app.cell
def _(Path, TORCH_MISMATCH, mo, torch):
    # (1) Tên pipeline, chế độ chạy, đường dẫn đầu ra (RunPaths), logger
    mo.stop(
        TORCH_MISMATCH,
        mo.md(
            "**Phải khởi động lại kernel**: bộ nhớ đang giữ `torch` khác bản trên đĩa "
            "(thường do vừa cài gói làm nâng torch), nên import `torchvision`/`timm` sẽ lỗi. "
            "Khởi động lại kernel rồi chạy lại (Run all)."
        ),
    )
    from src.utils.logger import setup_logger
    from src.utils.paths import RunPaths

    PIPELINE_NAME = "frcrn-sweep"
    RUN_TRAINING = True  # True: huấn luyện thật, ghi vào output/; False: smoke nhỏ ghi vào output/test
    RUN_TIMING = False  # Bước 1: đo thời gian 1 epoch smoke
    RUN_SWEEP = False  # Bước 2: khởi chạy sweep W&B (cần WANDB_API_KEY)
    RUN_CONFIRM = True  # Bước 3: huấn luyện cấu hình tốt nhất, 100 epoch (CONFIRM_SETTINGS)
    USE_WANDB = True  # theo dõi bằng W&B nếu có WANDB_API_KEY; thiếu key thì chạy không W&B
    USE_DRIVE = True  # sao lưu best.pt/resume.pt/result.json lên Drive để resume khi bị ngắt
    RESUME_RUN_ID = "frcrn-sweep-2026-10-16-11-37-49"  # lần chạy 100 epoch đang chạy: Run all sẽ gắn lại, không chạy trùng. Đặt None để chạy mới

    output_dir = Path("output") if RUN_TRAINING else Path("output") / "test"
    if RESUME_RUN_ID:
        run = RunPaths.from_run_id(RESUME_RUN_ID, output_dir)
    else:
        run = RunPaths.new(PIPELINE_NAME if RUN_TRAINING else f"{PIPELINE_NAME}-smoke", output_dir)
    run.create()

    logger = setup_logger(PIPELINE_NAME, run.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run %s | device %s | RUN_TRAINING=%s", run.run_id, device, RUN_TRAINING)
    return (
        RUN_CONFIRM,
        RUN_SWEEP,
        RUN_TIMING,
        RUN_TRAINING,
        USE_DRIVE,
        USE_WANDB,
        device,
        logger,
        run,
    )


@app.cell
def _(RUN_TRAINING):
    # (2) Tham số cấu hình: đọc từ config/frcrn_sweep.yaml; smoke giới hạn số mẫu
    from scripts.sweep_frcrn import load_sweep_config

    SWEEP = load_sweep_config()
    HIGH_SNR_LEVELS = tuple(float(x) for x in SWEEP["metrics"]["high_snr_levels"])
    REPORT_SNR_LEVELS = tuple(float(x) for x in SWEEP["metrics"]["report_snr_levels"])
    NUM_WORKERS = int(SWEEP["trial"]["num_workers"]) if RUN_TRAINING else 0
    KERNEL_WORKERS = 0  # đánh giá trong kernel marimo: không fork worker (huấn luyện chạy ở tiến trình riêng)
    MAX_SAMPLES = SWEEP["trial"]["max_samples"] if RUN_TRAINING else 64
    SWEEP_COUNT = int(SWEEP["trial"]["count"])
    CONFIRM_SEEDS = tuple(SWEEP["trial"]["seeds"])
    PERCEPTUAL_BATCHES = int(SWEEP["metrics"]["perceptual_max_batches"])

    # Cấu hình tốt nhất của sweep W&B wzzdup75 (lượt xqrcbdzg, noise_si_snr_hi -0,21 dB sau 10 epoch)
    CONFIRM_SETTINGS = {
        "lr": 0.000273365064091293,
        "lr_decay": 1.0,
        "halve_lr_after": 2,
        "batch_size": 12,
        "segment_seconds": 2.0,
        "grad_clip": 5.0,
        "mask_weight": 4.0,
        "target": "speech",
        "epochs": int(SWEEP["trial"]["confirm_epochs"]),
        "patience": int(SWEEP["trial"]["confirm_patience"]),
    }
    return (
        CONFIRM_SEEDS,
        CONFIRM_SETTINGS,
        HIGH_SNR_LEVELS,
        KERNEL_WORKERS,
        MAX_SAMPLES,
        NUM_WORKERS,
        PERCEPTUAL_BATCHES,
        REPORT_SNR_LEVELS,
        SWEEP,
        SWEEP_COUNT,
    )


@app.cell
def _(mo, os):
    # Token Kaggle: ưu tiên biến môi trường KAGGLE_ACCESS_TOKEN, nếu thiếu thì nhập vào ô này (không lưu vào file)
    kaggle_token = mo.ui.text(kind="password", label="KAGGLE_ACCESS_TOKEN (chỉ cần khi môi trường chưa có)")
    mo.md("`KAGGLE_ACCESS_TOKEN` đã có trong môi trường.") if os.environ.get("KAGGLE_ACCESS_TOKEN") else kaggle_token
    return (kaggle_token,)


@app.cell
def _(Path, kaggle_token, mo, os):
    # (3) Nạp dataset: tải `mix-dataset` từ Kaggle vào datasets/ (đã giải nén thì bỏ qua, không cần token)
    from scripts.download_dataset import DATASETS, ensure_dataset

    DATASET_NAME = "mixture-36l"
    extract_dir = Path("datasets") / DATASETS[DATASET_NAME].name
    if not (extract_dir.is_dir() and any(extract_dir.iterdir())) and not os.environ.get("KAGGLE_ACCESS_TOKEN"):
        mo.stop(not kaggle_token.value, mo.md("Nhập `KAGGLE_ACCESS_TOKEN` ở ô phía trên rồi chạy lại cell này."))
        os.environ["KAGGLE_ACCESS_TOKEN"] = kaggle_token.value

    dataset_root = ensure_dataset(DATASET_NAME, Path("datasets"))
    mo.md(f"`{DATASET_NAME}`: `{dataset_root}`")
    return (dataset_root,)


@app.cell
def _(dataset_root, mo):
    # (4) Pretrain: FRCRN_SE_16K gốc (HuggingFace, 161 MB) và best.pt đã fine-tune trước đó (để so mốc)
    from scripts.load_pretrain import download_checkpoint, resolve_url
    from scripts.sweep_frcrn import ORIGINAL_CHECKPOINT

    assert dataset_root.is_dir()
    if not ORIGINAL_CHECKPOINT.is_file():
        download_checkpoint(resolve_url("frcrn-pretrain"), ORIGINAL_CHECKPOINT.parent)
    PREVIOUS_BEST = ORIGINAL_CHECKPOINT.parent / "FRCRN" / "best.pt"
    mo.md(
        f"Gốc: `{ORIGINAL_CHECKPOINT}` | đã fine-tune trước đó: `{PREVIOUS_BEST}` (tồn tại: {PREVIOUS_BEST.is_file()})"
    )
    return ORIGINAL_CHECKPOINT, PREVIOUS_BEST


@app.cell
def _(RUN_CONFIRM, RUN_SWEEP, RUN_TIMING, SWEEP, USE_DRIVE, mo):
    # Sao lưu Drive: mỗi lượt/seed một thư mục con mới (<run_id>-...), không đụng FRCRN, FRCRN_vlsp, kd-moe-ft
    from scripts.sweep_frcrn import check_drive_dir

    DRIVE_DIR = check_drive_dir(SWEEP["drive"]["dir"])
    drive_fs = None
    if USE_DRIVE and (RUN_TIMING or RUN_SWEEP or RUN_CONFIRM):
        from gdrive_fsspec import GoogleDriveFileSystem as _DriveFileSystem

        drive_fs = _DriveFileSystem(
            use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False}
        )
    mo.md(
        f"Drive: `{DRIVE_DIR}` ({'đã kết nối' if drive_fs is not None else 'chưa kết nối: chỉ cần khi bật RUN_SWEEP/RUN_CONFIRM'})"
    )
    return DRIVE_DIR, drive_fs


@app.cell
def _(ORIGINAL_CHECKPOINT, mo):
    # (5) Dựng kiến trúc (hai UNet xếp tầng, kiến trúc không đổi) và đếm tham số
    from scripts.sweep_frcrn import build_trial_separator

    architecture = build_trial_separator(ORIGINAL_CHECKPOINT)
    mo.md(
        f"FRCRN: {architecture.num_parameters() / 1e6:.2f}M tham số | target `{architecture.config.target}` | "
        f"fft {architecture.config.fft_len}, win {architecture.config.win_len}, hop {architecture.config.win_inc}"
    )
    return (architecture,)


@app.cell
def _(HIGH_SNR_LEVELS, KERNEL_WORKERS, MAX_SAMPLES, ORIGINAL_CHECKPOINT, PREVIOUS_BEST, device, mo, run):
    # Bước 0: mốc, SI-SNR tiếng nói / nhiễu / SI-SNRi theo từng SNR (chỉ đánh giá, không huấn luyện)
    import json

    from scripts.sweep_frcrn import build_trial_separator as _build
    from scripts.sweep_frcrn import evaluate_by_snr, validation_loader

    _loader = validation_loader(max_samples=MAX_SAMPLES, num_workers=KERNEL_WORKERS)
    baseline_scores = {}
    for _name, _path in (("fe_se_16k_goc", ORIGINAL_CHECKPOINT), ("best_truoc_do", PREVIOUS_BEST)):
        if _path.is_file():
            baseline_scores[_name] = evaluate_by_snr(_build(_path), _loader, device, HIGH_SNR_LEVELS)
    (run.result_dir / "baseline_by_snr.json").write_text(json.dumps(baseline_scores, indent=2), encoding="utf-8")
    mo.ui.table(
        [{"checkpoint": _n, **{k: round(v, 2) for k, v in _s.items()}} for _n, _s in baseline_scores.items()],
        selection=None,
    )
    return (baseline_scores,)


@app.cell
def _(MAX_SAMPLES, NUM_WORKERS, ORIGINAL_CHECKPOINT, RUN_TIMING, device, logger, mo, run):
    # Bước 1: đo thời gian 1 epoch smoke (cờ RUN_TIMING), ngoại suy thô lên tập train
    import time

    from scripts.load_dataset import Mixture36Dataset
    from scripts.sweep_frcrn import run_trial as _run_trial

    mo.stop(not RUN_TIMING, mo.md("`RUN_TIMING = False`: bỏ qua bước đo thời gian."))
    _start = time.perf_counter()
    _run_trial(
        {"epochs": 1, "patience": 1},
        device=device,
        trial_dir=run.checkpoint_dir / "timing",
        checkpoint=ORIGINAL_CHECKPOINT,
        max_samples=MAX_SAMPLES,
        num_workers=NUM_WORKERS,
        smoke=True,
    )
    _elapsed = time.perf_counter() - _start
    _train_size = len(Mixture36Dataset(split="train", duration=2.0, load_noise=True))
    _per_epoch_minutes = _elapsed / (MAX_SAMPLES or 64) * _train_size / 60
    logger.info("timing: %.1fs cho %d mẫu smoke", _elapsed, MAX_SAMPLES or 64)
    mo.md(
        f"Ước lượng thô: **{_per_epoch_minutes:.1f} phút/epoch** trên {_train_size} mẫu (chưa kiểm chứng; gồm cả đánh giá)."
    )
    return


@app.cell
def _(
    HIGH_SNR_LEVELS,
    MAX_SAMPLES,
    NUM_WORKERS,
    ORIGINAL_CHECKPOINT,
    RUN_SWEEP,
    RUN_TRAINING,
    SWEEP,
    SWEEP_COUNT,
    DRIVE_DIR,
    device,
    drive_fs,
    logger,
    mo,
    os,
    run,
):
    # Bước 2: sweep W&B (cờ RUN_SWEEP). Cần WANDB_API_KEY trong môi trường
    mo.stop(not RUN_SWEEP, mo.md("`RUN_SWEEP = False`: chưa khởi chạy sweep."))
    mo.stop(not os.environ.get("WANDB_API_KEY"), mo.md("Đặt biến môi trường `WANDB_API_KEY` rồi chạy lại cell này."))
    import wandb

    from scripts.sweep_frcrn import make_wandb_trial

    sweep_id = wandb.sweep(SWEEP["sweep"], project=SWEEP["wandb"]["project"])
    logger.info("W&B sweep %s, %d lượt", sweep_id, SWEEP_COUNT)
    wandb.agent(
        sweep_id,
        function=make_wandb_trial(
            device=device,
            root_dir=run.checkpoint_dir / "sweep",
            checkpoint=ORIGINAL_CHECKPOINT,
            high_snr_levels=HIGH_SNR_LEVELS,
            max_samples=MAX_SAMPLES,
            num_workers=NUM_WORKERS,
            smoke=not RUN_TRAINING,
            fs=drive_fs,
            drive_dir=DRIVE_DIR,
        ),
        project=SWEEP["wandb"]["project"],
        count=SWEEP_COUNT,
    )
    mo.md(f"Sweep `{sweep_id}` xong. Chọn cấu hình tốt nhất trong W&B rồi điền vào `CONFIRM_SETTINGS`.")
    return


@app.cell
def _(
    CONFIRM_SEEDS,
    CONFIRM_SETTINGS,
    DRIVE_DIR,
    HIGH_SNR_LEVELS,
    MAX_SAMPLES,
    NUM_WORKERS,
    ORIGINAL_CHECKPOINT,
    Path,
    RUN_CONFIRM,
    RUN_TRAINING,
    SWEEP,
    USE_DRIVE,
    USE_WANDB,
    mo,
    os,
    run,
):
    # Bước 3: huấn luyện chạy ở tiến trình riêng (DataLoader worker trong kernel marimo từng thoát đột ngột).
    # Cell không chờ: khởi chạy hoặc gắn lại tiến trình đang chạy rồi trả về ngay (kernel marimo chỉ chạy một việc một lúc).
    # Khi huấn luyện xong, chạy lại cell này để đọc kết quả; RESUME_RUN_ID tiếp tục từ resume.pt (local hoặc Drive).
    from scripts.sweep_frcrn import best_epoch_scores, launch_training_process, read_training_result, training_pid

    _tracked = USE_WANDB and bool(os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").is_file())
    confirm_results = {}
    _running = []
    for _seed in CONFIRM_SEEDS if RUN_CONFIRM else ():
        _stage = f"{run.run_id}-confirm-seed{_seed}"
        _work = run.checkpoint_dir / _stage
        _name = f"{run.run_id}-seed{_seed}"
        _spec = {
            "settings": {**CONFIRM_SETTINGS, "seed": _seed},
            "trial_dir": str(run.checkpoint_dir),
            "stage_name": _stage,
            "checkpoint": str(ORIGINAL_CHECKPOINT),
            "high_snr_levels": list(HIGH_SNR_LEVELS),
            "max_samples": MAX_SAMPLES,
            "num_workers": NUM_WORKERS,
            "smoke": not RUN_TRAINING,
            "use_drive": USE_DRIVE,
            "drive_dir": DRIVE_DIR,
        }
        if _tracked:
            _spec.update(wandb_project=SWEEP["wandb"]["final_project"], run_name=_name, run_id=_name)
        launch_training_process(_spec, _work / "proc")
        _pid = training_pid(_work / "proc" / "train.pid")
        if _pid is not None:
            _running.append(f"`{_stage}` (pid {_pid}, log `{_work / 'proc' / 'train.log'}`)")
            continue
        _result = read_training_result(_work / "result.json", _work / "proc")
        confirm_results[f"seed{_seed}"] = {"noise_si_snr_hi": _result["best_score"], **best_epoch_scores(_result)}
    mo.md(
        "`RUN_CONFIRM = False`: chưa huấn luyện cấu hình xác nhận."
        if not RUN_CONFIRM
        else (
            (
                "Đang huấn luyện ở tiến trình riêng: "
                + "; ".join(_running)
                + ". Chạy lại cell này khi xong để lấy kết quả. "
            )
            if _running
            else ""
        )
        + f"W&B: {'bật' if _tracked else 'tắt (thiếu khoá W&B hoặc USE_WANDB=False)'}"
    )
    return (confirm_results,)


@app.cell
def _(
    HIGH_SNR_LEVELS,
    KERNEL_WORKERS,
    MAX_SAMPLES,
    PERCEPTUAL_BATCHES,
    REPORT_SNR_LEVELS,
    baseline_scores,
    confirm_results,
    device,
    mo,
    run,
):
    # Bảng so sánh theo từng SNR (-5 đến 20 dB) và PESQ/STOI cho cấu hình xác nhận
    from scripts.sweep_frcrn import build_trial_separator as _build_separator
    from scripts.sweep_frcrn import perceptual_scores, sweep_table
    from scripts.sweep_frcrn import validation_loader as _validation_loader

    _rows = {name: dict(s) for name, s in baseline_scores.items()}
    _rows.update(confirm_results)
    _table = sweep_table(_rows, REPORT_SNR_LEVELS)

    _first = next(iter(confirm_results), None)
    _checkpoint = run.checkpoint_dir / f"{run.run_id}-confirm-{_first}" / "best.pt" if _first else None
    _perceptual = {}
    if _checkpoint is not None and _checkpoint.is_file():
        _separator = _build_separator(_checkpoint)
        _loader = _validation_loader(max_samples=MAX_SAMPLES, num_workers=KERNEL_WORKERS)
        _perceptual = perceptual_scores(_separator, _loader, device, PERCEPTUAL_BATCHES, HIGH_SNR_LEVELS)
    mo.vstack(
        [
            mo.md("SI-SNR nhiễu (dB) theo từng SNR; xếp theo trung bình 15 và 20 dB."),
            mo.ui.table(
                [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in _table],
                selection=None,
            ),
            mo.md(f"PESQ/STOI (cấu hình xác nhận): `{ {k: round(v, 3) for k, v in _perceptual.items()} }`"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
