import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import torch

    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return Path, mo, os, torch


@app.cell
def _(mo):
    mo.md(r"""
    # KD teacher-student với ReviewKD trên Mixture-36l

    | Nhánh | Đường đi | Trạng thái |
    | --- | --- | --- |
    | Teacher | mixture → FRCRN (nhiễu ước lượng, chuẩn hoá RMS) → BEATs (lớp 3, 6, 9, 12) | đóng băng |
    | Student | mixture → log-mel → PANNs CNN10 (4 stage) → 36 lớp | huấn luyện từ đầu |
    | KD | ReviewKD: ABF hợp nhất stage sâu vào stage nông, HCL (MSE + pyramid 4/2/1) với 4 lớp BEATs | chỉ lúc train |

    Loss: $\mathcal{L}=\mathcal{L}_{CE}+\lambda\,w(e)\,\sum_{i=1}^{4}\mathrm{HCL}(\mathrm{ABF}_i,\,F_t^{i})$,
    với $w(e)$ tăng tuyến tính trong `review_warmup_epochs` epoch đầu (như mã gốc). Chạy $\lambda=0$ cho baseline CE thuần
    (cùng dữ liệu, cùng lịch học, cùng seed) để so sánh công bằng.

    Cấu hình: `config/reviewkd.yaml`. Mã: `src/model/reviewkd/`, `src/loss/reviewkd.py`,
    `src/train/reviewkd_trainer.py`, `scripts/run_reviewkd.py`. Chi tiết toán: `latex/reviewkd/main.tex`.

    **Chạy**: đặt `RUN_TRAINING = True` rồi chọn **Run all**. `False` chạy smoke test nhỏ (1 epoch, 64 clip) vào `output/test/`.
    Cell cuối đặt `is_completed = True` chỉ khi `RUN_TRAINING = True` và mọi bước đã xong.
    """)
    return


@app.cell
def _(Path, torch):
    # (1) Tên pipeline, chế độ chạy, đường dẫn đầu ra (RunPaths), logger
    from src.utils.logger import setup_logger
    from src.utils.paths import RunPaths

    PIPELINE_NAME = "reviewkd"
    RUN_TRAINING = False  # True: chạy thật; False: smoke test nhỏ ghi vào output/test
    RESUME_RUN_ID = None  # ví dụ "reviewkd-2026-10-17-09-00-00": mô hình đã có history.json thì bỏ qua

    output_dir = Path("output") if RUN_TRAINING else Path("output") / "test"
    if RESUME_RUN_ID:
        run = RunPaths.from_run_id(RESUME_RUN_ID, output_dir)
    else:
        run = RunPaths.new(PIPELINE_NAME if RUN_TRAINING else f"{PIPELINE_NAME}-smoke", output_dir)
    run.create()

    logger = setup_logger(PIPELINE_NAME, run.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run %s | device %s | RUN_TRAINING=%s", run.run_id, device, RUN_TRAINING)
    return RUN_TRAINING, device, logger, run


@app.cell
def _():
    # (2) Tham số cấu hình: số worker, có chạy baseline CE thuần (lambda = 0) không, dùng Drive để lấy FRCRN không
    NUM_WORKERS = 8
    INCLUDE_BASELINE = True
    USE_DRIVE = False  # True: nếu thiếu pretrained/frcrn_best.pt thì tải từ Google Drive (cần gdrive_fsspec)
    return INCLUDE_BASELINE, NUM_WORKERS, USE_DRIVE


@app.cell
def _(mo, os):
    # Token Kaggle: ưu tiên biến môi trường KAGGLE_ACCESS_TOKEN, nếu thiếu thì nhập vào ô này (không lưu vào file)
    kaggle_token = mo.ui.text(kind="password", label="KAGGLE_ACCESS_TOKEN (chỉ cần khi môi trường chưa có)")
    mo.md("`KAGGLE_ACCESS_TOKEN` đã có trong môi trường.") if os.environ.get("KAGGLE_ACCESS_TOKEN") else kaggle_token
    return (kaggle_token,)


@app.cell
def _(Path, kaggle_token, mo, os):
    # (3) Nạp dataset: Mixture-36l (train/validation); AudioSet-40l chỉ cần khi target_rms = 0 (tính lại RMS nhiễu)
    from scripts.download_dataset import DATASETS, ensure_dataset
    from scripts.run_reviewkd import read_reviewkd_config

    raw = read_reviewkd_config()
    wanted = [raw["dataset"]["name"]]
    if not raw["review"]["target_rms"]:
        wanted.append(raw["dataset"]["noise_dataset"])
    missing = [
        key
        for key in wanted
        if not (
            (Path("datasets") / DATASETS[key].name).is_dir() and any((Path("datasets") / DATASETS[key].name).iterdir())
        )
    ]
    if missing and not os.environ.get("KAGGLE_ACCESS_TOKEN"):
        mo.stop(not kaggle_token.value, mo.md("Nhập `KAGGLE_ACCESS_TOKEN` ở ô phía trên rồi chạy lại cell này."))
        os.environ["KAGGLE_ACCESS_TOKEN"] = kaggle_token.value

    dataset_roots = {key: ensure_dataset(key, Path("datasets")) for key in wanted}
    mo.md("\n".join(f"- `{key}`: `{root}`" for key, root in dataset_roots.items()))
    return dataset_roots, raw


@app.cell
def _(USE_DRIVE, mo, raw):
    # (4) Pretrain: BEATs thuần từ HuggingFace, FRCRN từ pretrained/ (hoặc Drive); cả hai bị đóng băng khi train
    from scripts.load_pretrain import download_frcrn_best, download_pretrained
    from scripts.run_reviewkd import resolve_path

    beats_path = resolve_path(raw["teacher"]["beats_checkpoint"])
    if not beats_path.is_file():
        download_pretrained("beats")
    frcrn_path = resolve_path(raw["teacher"]["frcrn_checkpoint"])
    if not frcrn_path.is_file():
        if not USE_DRIVE:
            raise FileNotFoundError(f"{frcrn_path} not found: copy FRCRN best.pt there or set USE_DRIVE = True")
        from gdrive_fsspec import GoogleDriveFileSystem

        _fs = GoogleDriveFileSystem(
            use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False}
        )
        download_frcrn_best(_fs, frcrn_path)
    mo.md(f"FRCRN: `{frcrn_path}`  \nBEATs: `{beats_path}`")
    return beats_path, frcrn_path


@app.cell
def _(beats_path, frcrn_path, mo, raw):
    # (5) Kiến trúc: lưới đặc trưng của student và teacher phải khớp, đếm tham số
    from scripts.run_baselines import count_millions
    from scripts.run_reviewkd import build_review, build_student, check_grids
    from src.model.reviewkd import ReviewKDConfig

    assert frcrn_path.is_file() and beats_path.is_file()
    grid = check_grids(raw)
    review_config = ReviewKDConfig.from_dict(raw["review"])
    student_params = count_millions(build_student(raw))
    review_params = count_millions(build_review(raw, review_config, grid))
    mo.md(
        f"Lưới đặc trưng (tần số x thời gian): **{grid}** | student {student_params:.2f}M tham số | "
        f"module ReviewKD {review_params:.2f}M (chỉ lúc train) | lớp BEATs {list(review_config.teacher_layers)}"
    )
    return (grid,)


@app.cell
def _(INCLUDE_BASELINE, NUM_WORKERS, RUN_TRAINING, dataset_roots, device, grid, logger, run):
    # (6) Huấn luyện: baseline CE (lambda = 0) rồi ReviewKD; mô hình đã có history.json thì bỏ qua.
    # dataset_roots và grid là tham số để marimo chỉ chạy cell này sau khi dữ liệu và lưới đã sẵn sàng.
    from scripts.run_reviewkd import run_reviewkd

    assert dataset_roots and grid
    rows = run_reviewkd(
        run,
        device,
        logger,
        include_baseline=INCLUDE_BASELINE,
        smoke=not RUN_TRAINING,
        num_workers=NUM_WORKERS,
    )
    return (rows,)


@app.cell
def _(mo, rows):
    # Bảng kết quả validation: macro_acc (trung bình balanced accuracy qua các SNR) và 9 metrics
    from scripts.run_baselines import markdown_table
    from scripts.run_reviewkd import RESULT_COLUMNS

    mo.md(markdown_table(rows, RESULT_COLUMNS))
    return


@app.cell
def _(INCLUDE_BASELINE, RUN_TRAINING, mo, rows, run):
    # Hoàn thành: is_completed = True chỉ khi chạy thật (RUN_TRAINING), mọi mô hình có history, result.csv đã ghi
    import json

    expected = ["cnn10-reviewkd"] + (["cnn10-ce"] if INCLUDE_BASELINE else [])
    artefacts_ready = (
        all(run.history_path(name).is_file() for name in expected)
        and run.result_file.is_file()
        and {row["model"] for row in rows} == set(expected)
    )
    is_completed = RUN_TRAINING and artefacts_ready
    marker = run.result_dir / "is_completed.json"
    marker.write_text(
        json.dumps(
            {
                "is_completed": is_completed,
                "mode": "full" if RUN_TRAINING else "smoke",
                "artefacts_ready": artefacts_ready,
                "models": expected,
            }
        ),
        encoding="utf-8",
    )
    mo.md(f"`is_completed = {is_completed}` ({'full' if RUN_TRAINING else 'smoke'}) — `{marker}`")
    return (is_completed,)


if __name__ == "__main__":
    app.run()
