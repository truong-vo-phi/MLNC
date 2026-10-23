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
    # Baseline CNN / RNN / ViT / lai trên Mixture-36l (mix-dataset)

    16 baseline có 0,7-28M tham số, huấn luyện **từ đầu** trên cùng log-mel 128 x 384 (4 s), cùng SpecAugment,
    cùng optimizer, lịch học và tiêu chí chọn (`macro_f1` trên validation), rồi đánh giá bằng **9 metrics**.

    | Họ | Mô hình |
    | --- | --- |
    | CNN | MobileNetV3-L (width 0,4 / 0,5 / 1,0 / 2,0), PANNs CNN10, ResNet18, EfficientNet-B2, ConvNeXt V2-T |
    | RNN | CRNN (BiGRU), BiLSTM |
    | ViT | DeiT-Tiny, ViT-Mini, ViT-S/16 |
    | Lai | MobileViT-S, Conformer-M, Swin-T |

    Tên mô hình trong bảng kết quả ghi rõ **kiến trúc**, không phải trọng số: không có checkpoint EfficientAT,
    PANNs, CED hay HTS-AT nào được nạp. Mã nguồn nằm ở `src/`, `scripts/`, cấu hình ở `config/baselines.yaml`.

    **Chạy**: nhập `KAGGLE_ACCESS_TOKEN` ở cell tải dataset (nếu môi trường chưa có), rồi chọn **Run all**.
    """)
    return


@app.cell
def _(Path, torch):
    # (1) Tên pipeline, chế độ chạy, đường dẫn đầu ra (RunPaths), logger
    from src.utils.logger import setup_logger
    from src.utils.paths import RunPaths

    PIPELINE_NAME = "baselines"
    RUN_TRAINING = False  # True: chạy thật; False: smoke test nhỏ ghi vào output/test (molab đặt True)
    RESUME_RUN_ID = None  # ví dụ "baselines-2026-10-14-21-50-17" để chạy tiếp lần chạy cũ

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
    # (2) Tham số cấu hình: dataset (khớp config/baselines.yaml), tập baseline cần chạy, số worker
    DATASET_NAME = "mixture-36l"
    DURATION = 4.0
    NUM_FRAMES = 384
    SELECTED_MODELS = None  # None = cả 16 baseline
    NUM_WORKERS = 8
    CONFUSION_MODEL = None  # None: model có macro_f1 cao nhất
    return (
        CONFUSION_MODEL,
        DATASET_NAME,
        DURATION,
        NUM_FRAMES,
        NUM_WORKERS,
        SELECTED_MODELS,
    )


@app.cell
def _(mo, os):
    # Token Kaggle: ưu tiên biến môi trường KAGGLE_ACCESS_TOKEN, nếu thiếu thì nhập vào ô này (không lưu vào file)
    kaggle_token = mo.ui.text(kind="password", label="KAGGLE_ACCESS_TOKEN (chỉ cần khi môi trường chưa có)")
    mo.md("`KAGGLE_ACCESS_TOKEN` đã có trong môi trường.") if os.environ.get("KAGGLE_ACCESS_TOKEN") else kaggle_token
    return (kaggle_token,)


@app.cell
def _(DATASET_NAME, Path, kaggle_token, mo, os):
    # (3) Nạp dataset: tải `mix-dataset` từ Kaggle vào datasets/ (đã giải nén thì bỏ qua, không cần token)
    from scripts.download_dataset import DATASETS, ensure_dataset

    extract_dir = Path("datasets") / DATASETS[DATASET_NAME].name
    if not (extract_dir.is_dir() and any(extract_dir.iterdir())) and not os.environ.get("KAGGLE_ACCESS_TOKEN"):
        mo.stop(not kaggle_token.value, mo.md("Nhập `KAGGLE_ACCESS_TOKEN` ở ô phía trên rồi chạy lại cell này."))
        os.environ["KAGGLE_ACCESS_TOKEN"] = kaggle_token.value

    dataset_root = ensure_dataset(DATASET_NAME, Path("datasets"))
    mo.md(f"`{DATASET_NAME}`: `{dataset_root}`")
    return (dataset_root,)


@app.cell
def _(DATASET_NAME, DURATION, NUM_FRAMES, dataset_root, mo, run):
    # Cấu hình baseline theo dataset đã chọn (ghi cạnh kết quả) và danh sách lớp
    from scripts.load_dataset import build_dataset
    from scripts.run_baselines import write_variant_config

    assert dataset_root.is_dir()
    config_path = write_variant_config(
        run.result_dir / "baselines.yaml", dataset_name=DATASET_NAME, duration=DURATION, num_frames=NUM_FRAMES
    )
    class_names = build_dataset(DATASET_NAME, "train", duration=DURATION).class_names
    mo.md(f"{len(class_names)} lớp | clip {DURATION} s | {NUM_FRAMES} frame | cấu hình: `{config_path}`")
    return class_names, config_path


@app.cell
def _(mo):
    mo.md("""
    **(4) Pretrain**: không nạp checkpoint pretrain. Checkpoint AudioSet đã thấy nhãn của các clip nguồn (rò rỉ) "
        "và dùng front-end 32 kHz / 64 mel khác dữ liệu 16 kHz. Cờ `pretrained` trong `config/baselines.yaml` "
        "chỉ bật được cho các kiến trúc timm (khởi tạo ImageNet).
    """)
    return


@app.cell
def _(class_names, config_path, mo):
    # (5) Dựng kiến trúc và đếm tham số thực tế với đầu ra đúng số lớp
    from scripts.run_baselines import describe_baselines

    architectures = describe_baselines(config_path, num_classes=len(class_names))
    mo.ui.table(architectures, selection=None)
    return


@app.cell
def _(
    NUM_WORKERS,
    RUN_TRAINING,
    SELECTED_MODELS,
    config_path,
    device,
    logger,
    run,
):
    # (6) Huấn luyện từng baseline (bỏ qua model đã xong), ghi result.csv sau mỗi model
    from scripts.run_baselines import run_baselines

    rows = run_baselines(
        run,
        device,
        logger,
        names=SELECTED_MODELS,
        config_path=config_path,
        num_workers=NUM_WORKERS,
        smoke=not RUN_TRAINING,
    )
    return (rows,)


@app.cell
def _(mo, rows, run):
    # Bảng so sánh 16 baseline theo 9 metrics (xếp theo macro_f1), ghi comparison.csv
    from scripts.finetune import METRIC_NAMES
    from scripts.run_baselines import best_models, comparison_table, write_comparison

    comparison_path = write_comparison(rows, run)
    leaders = best_models(rows)
    leader_lines = "\n".join(f"- `{metric}`: **{leaders[metric]}**" for metric in METRIC_NAMES)
    mo.vstack(
        [
            mo.md(f"Bảng so sánh (validation, xếp theo `macro_f1`): `{comparison_path}`"),
            mo.ui.table(comparison_table(rows), selection=None),
            mo.md(f"Model tốt nhất theo từng metric:\n\n{leader_lines}"),
        ]
    )
    return (leaders,)


@app.cell
def _(CONFUSION_MODEL, class_names, leaders, mo, run):
    # Vẽ confusion matrix (tỉ lệ trên lớp thật) của từng baseline vào output/figure/<run>/
    from scripts.run_baselines import load_confusions, plot_baseline_confusions

    figure_paths = plot_baseline_confusions(load_confusions(run), class_names, run)
    model_picker = mo.ui.dropdown(
        options=sorted(figure_paths),
        value=CONFUSION_MODEL or leaders["macro_f1"],
        label="Confusion matrix của model",
    )
    mo.output.replace(model_picker)
    return figure_paths, model_picker


@app.cell
def _(figure_paths, mo, model_picker):
    mo.image(src=str(figure_paths[model_picker.value]))
    return


if __name__ == "__main__":
    app.run()
