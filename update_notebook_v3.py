import marimo._code_mode as cm

pkri_code = '''# (1) Cấu hình chế độ thực thi
import torch
from ced_tiny.runtime import resolve_device

from src.utils.logger import setup_logger
from src.utils.paths import RunPaths

PIPELINE_NAME = "ced-tiny-v3"
RUN_TRAINING = True  # False: Smoke test nhanh; True: Huấn luyện đầy đủ trên tập dữ liệu hoàn chỉnh
RESUME_RUN_ID = None
DEVICE_PREF = None  # "cuda:0", "cpu" hoặc None (tự động phát hiện)

device = resolve_device(DEVICE_PREF)
output_base = Path("output") if RUN_TRAINING else Path("output/test")
run = RunPaths.from_run_id(RESUME_RUN_ID, output_base) if RESUME_RUN_ID else RunPaths.new(PIPELINE_NAME, output_base)
run.log_file.parent.mkdir(parents=True, exist_ok=True)
logger = setup_logger(PIPELINE_NAME, run.log_file)

print(f"Run ID    : {run.run_id}")
print(f"Device    : {device}")
print(f"Mode      : {'HUẤN LUYỆN TOÀN DIỆN (CED-Tiny Cải Tiến v3)' if RUN_TRAINING else 'SMOKE TEST'}")
print(f"Log File  : {run.log_file}")'''

sfpl_code = '''# (2) Cấu hình chi tiết các thành phần (Dễ dàng tùy biến)
from ced_tiny.config import (
    CEDDataConfig,
    CEDTrainConfig,
    FrontendConfig,
    SpecAugmentConfig,
    load_config,
)

from src.utils.seed import set_seed

CONFIG_YAML = PACKAGE_ROOT / "config" / "ced_tiny.yaml"
cfg = load_config(CONFIG_YAML)

# ==============================================================================
# CẤU HÌNH CED-TINY CẢI TIẾN V3:
# ==============================================================================
MODEL_TITLE = "CED-Tiny Cải Tiến v3 (2D Factorized Bias + Pure CLS + Zero-Shock Init)"
USE_BIAS = True
TIME_MAJOR = False  # Giữ trật tự Freq-major gốc khớp 100% AudioSet
HYBRID_POOL = False  # Pure CLS Pooling: chọn lọc động các khung nhiễu sạch ở SNR cao
BIAS_INIT = "zero"  # Zero-Shock Identity Init: bảo toàn 100% Attention gốc tại Step 0

# 1. Cấu hình FRCRN Separator
FRCRN_CHECKPOINT = "pretrained/FRCRN/best_e037_20261020T203958Z.pt"
TARGET_RMS = 0.07925
FREEZE_FRCRN = True

# 2. Cấu hình Frontend LogMel & SpecAugment
N_MELS = 64
NUM_FRAMES = 398
FREQ_MASKS = 2
FREQ_WIDTH = 8
TIME_MASKS = 2
TIME_WIDTH = 24

# 3. Cấu hình CED-Tiny Backbone
CED_ARCH = "deit_tiny_patch16_224"
NUM_CLASSES = 36
CED_INIT = "ced-tiny" if RUN_TRAINING else "none"
DROPOUT = 0.0

# 4. Cấu hình Dữ Liệu
DURATION = 4.0
SNR_LEVELS = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
BALANCED_SAMPLING = True
NHAT_CACHE_DIR = str(globals().get("cache_nhat_dir", "/marimo/dataset/cache"))
NUM_WORKERS = 0  # Đọc mmap trực tiếp siêu nhanh, an toàn với asyncio

# 5. Cấu hình Huấn Luyện
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32 if RUN_TRAINING else 4
ACCUMULATION_STEPS = 1
MAX_EPOCHS = 15 if RUN_TRAINING else 1
PATIENCE = 7
LABEL_SMOOTHING = 0.05
MIXUP = 0.5
MIXUP_ALPHA = 0.4
AMP_MODE = "bf16"
SEED = 2026

frontend_cfg = FrontendConfig(n_mels=N_MELS, num_frames=NUM_FRAMES)
specaugment_cfg = SpecAugmentConfig(
    freq_masks=FREQ_MASKS, freq_width=FREQ_WIDTH, time_masks=TIME_MASKS, time_width=TIME_WIDTH
)
data_cfg = CEDDataConfig(
    duration=DURATION,
    snr_levels=SNR_LEVELS,
    balanced_sampling=BALANCED_SAMPLING,
    num_workers=NUM_WORKERS,
    nhat_cache_dir=NHAT_CACHE_DIR,
    max_train_samples=None if RUN_TRAINING else 32,
    max_eval_samples=None if RUN_TRAINING else 16,
)
train_cfg = CEDTrainConfig(
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
    accumulation_steps=ACCUMULATION_STEPS,
    epochs=MAX_EPOCHS,
    patience=PATIENCE,
    label_smoothing=LABEL_SMOOTHING,
    mixup=MIXUP,
    mixup_alpha=MIXUP_ALPHA,
    amp=AMP_MODE,
    seed=SEED,
)

set_seed(train_cfg.seed)
print(f"Đã cấu hình thành công: {MODEL_TITLE}")
print(f"Epochs: {train_cfg.epochs} | Batch Size: {train_cfg.batch_size} | LR: {train_cfg.lr} | Mixup: {train_cfg.mixup}")
print(f"Use Bias: {USE_BIAS} (2D Factorized) | Time Major: {TIME_MAJOR} | Hybrid Pool: {HYBRID_POOL}")
print(f"Dataset Cache: {data_cfg.nhat_cache_dir} | Num Workers: {data_cfg.num_workers}")'''

zhcj_code = '''# (6) Huấn luyện mô hình CED-Tiny Cải Tiến v3
import time

from ced_tiny.trainer import CEDTinyTrainer

print("\\n" + "=" * 65)
print(f"BẮT ĐẦU HUẤN LUYỆN: {MODEL_TITLE.upper()}")
print("=" * 65)

ckpt_path = run.checkpoint_path("ced_tiny_v3")
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
resume_path = run.checkpoint_dir / "resume" / "ced_tiny_v3.pt"
resume_path.parent.mkdir(parents=True, exist_ok=True)

trainer = CEDTinyTrainer(
    model=model,
    config=train_cfg,
    device=device,
    logger=logger,
    name="ced_tiny_v3",
    checkpoint_path=ckpt_path,
    separator=None,
    target_rms=TARGET_RMS,
    resume_path=resume_path,
)

t0 = time.time()
fit_res = trainer.fit(loaders["train"], loaders["validation"])
elapsed = time.time() - t0

trainer.restore_best()
print("\\n" + "=" * 65)
print(f"HOÀN TẤT HUẤN LUYỆN TRONG {elapsed / 60:.2f} PHÚT!")
print(f"  • Best Epoch : {fit_res.best_epoch}")
print(f"  • Best Score : {fit_res.best_score:.4f} ({train_cfg.selection_metric})")
print("=" * 65)'''

qnkx_code = '''# (7) Đánh giá chi tiết & Xuất bảng 9 chỉ số
import pandas as pd
from ced_tiny.evaluate import (
    collect_predictions,
    compute_metrics,
    compute_snr_breakdown,
    compute_unseen_metrics,
    write_result_csv,
    write_snr_breakdown_csv,
)

print(f"Đang thu thập dự đoán cho {MODEL_TITLE}...")
preds = collect_predictions(
    model=model,
    loader=loaders["validation"],
    device=device,
    separator=None,
    target_rms=TARGET_RMS,
    amp_mode=train_cfg.amp,
    desc="Val v3",
)

overall = compute_metrics(preds.probs, preds.labels, model.config.num_classes)
snr_breakdown = compute_snr_breakdown(preds, data_cfg.snr_levels, model.config.num_classes)
unseen_overall, unseen_snr, n_unseen = compute_unseen_metrics(
    preds,
    data_cfg.audioset_eval_segments,
    data_cfg.audioset_train_segments,
    data_cfg.snr_levels,
    model.config.num_classes,
)

write_result_csv(run.result_dir / "result_v3.csv", [{"split": "validation", **overall}])
write_snr_breakdown_csv(run.result_dir / "snr_breakdown_v3.csv", snr_breakdown)

# 1. Bảng 9 chỉ số tổng thể
metric_names_vn = {
    "top1_accuracy": "Top-1 Accuracy",
    "top3_accuracy": "Top-3 Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision_macro": "Precision (Macro)",
    "recall_macro": "Recall (Macro)",
    "macro_f1": "Macro F1-Score",
    "micro_f1": "Micro F1-Score",
    "mAP": "mAP (Mean Average Precision)",
    "macro_auc": "Macro AUC",
}

report_rows = []
for _m_key, _m_name in metric_names_vn.items():
    _val = overall.get(_m_key, 0.0)
    report_rows.append({"Chỉ số Đánh Giá": _m_name, "Kết Quả v3": f"{_val:.4f} ({_val * 100:.2f}%)"})

df_metrics = pd.DataFrame(report_rows)
print("\\n" + "=" * 65)
print("BẢNG ĐÁNH GIÁ HIỆU NĂNG CED-TINY CẢI TIẾN V3 (9 CHỈ SỐ):")
print("=" * 65)
print(df_metrics.to_string(index=False))

# 2. Bảng theo 6 mức SNR
snr_rows = []
for _snr in sorted(data_cfg.snr_levels):
    _snr_int = int(_snr)
    _acc = snr_breakdown.get(f"acc_snr_{_snr_int}", 0.0)
    snr_rows.append({"SNR (dB)": _snr_int, "Top-1 Accuracy": f"{_acc:.4f} ({_acc * 100:.2f}%)"})

df_snr = pd.DataFrame(snr_rows)
print("\\n" + "=" * 65)
print("ĐỘ CHÍNH XÁC THEO 6 MỨC SNR (-5 dB ĐẾN 20 dB):")
print("=" * 65)
print(df_snr.to_string(index=False))'''

vxnm_code = '''# (8) Trực quan hóa đường cong huấn luyện & hiệu năng SNR
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

hist = trainer.history
if hist:
    epochs_list = [_h.get("epoch", _idx + 1) for _idx, _h in enumerate(hist)]
    t_loss = [_h.get("train_loss", 0.0) for _h in hist]
    v_acc = [_h.get("val_top1_accuracy", _h.get("top1_accuracy", 0.0)) for _h in hist]

    axes[0].plot(epochs_list, t_loss, "o-", color="#0284c7", label="Train Loss", linewidth=2.5)
    axes[0].set_title("Hàm Mất Mát Huấn Luyện (Loss Curve)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs_list, v_acc, "s-", color="#10b981", label="Validation Top-1 Acc", linewidth=2.5)
    axes[1].set_title("Độ Chính Xác Validation (Accuracy Curve)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Top-1 Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

plt.tight_layout()
curves_path = run.result_dir / "learning_curves_v3.png"
plt.savefig(curves_path, dpi=200)
print(f"Đã lưu biểu đồ học tập: {curves_path}")
plt.show()

# Biểu đồ SNR
plt.figure(figsize=(9, 4.5))
snr_vals = [_r["SNR (dB)"] for _r in snr_rows]
acc_vals = [float(_r["Top-1 Accuracy"].split()[0]) for _r in snr_rows]
plt.plot(snr_vals, acc_vals, "o-", color="#6366f1", linewidth=2.5, markersize=7)
plt.title("Độ Bền Âm Học Theo 6 Mức SNR (-5 dB đến 20 dB)", fontsize=13, fontweight="bold")
plt.xlabel("SNR (dB)", fontsize=11)
plt.ylabel("Top-1 Accuracy", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
snr_path = run.result_dir / "snr_performance_v3.png"
plt.savefig(snr_path, dpi=200)
print(f"Đã lưu biểu đồ SNR: {snr_path}")
plt.show()'''

ulza_code = '''# (9) Xuất checkpoint CED-Tiny Cải Tiến v3 độc lập cho triển khai
export_path = ROOT / "pretrained" / f"ced_tiny_v3_{run.run_id}.pt"
export_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {"model": model.state_dict(), "config": model.config.to_dict()},
    export_path
)
sz_mb = export_path.stat().st_size / 1e6

print("=" * 65)
print("XUẤT MÔ HÌNH THÀNH CÔNG CHO TRIỂN KHAI:")
print("=" * 65)
print(f"  • File path : {export_path}")
print(f"  • Kích thước: {sz_mb:.2f} MB")
print("=" * 65)'''

async with cm.get_context() as ctx:
    ctx.edit_cell("PKri", code=pkri_code)
    ctx.edit_cell("SFPL", code=sfpl_code)
    ctx.edit_cell("ZHCJ", code=zhcj_code)
    ctx.edit_cell("qnkX", code=qnkx_code)
    ctx.edit_cell("Vxnm", code=vxnm_code)
    ctx.edit_cell("ulZA", code=ulza_code)
print("All cells updated to CED-Tiny v3 successfully!")
