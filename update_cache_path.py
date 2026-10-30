import marimo._code_mode as cm

# Update Cell MJUe to point cache_nhat_dir to /marimo/dataset/cache
mjue_code = '''import pathlib as _pathlib

cache_nhat_dir = _pathlib.Path("/marimo/dataset/cache")
print(f"Dataset cache-nhat ready at: {cache_nhat_dir}")'''

# Update Cell SFPL
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
NHAT_CACHE_DIR = "/marimo/dataset/cache"
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

async with cm.get_context() as ctx:
    ctx.edit_cell("MJUe", code=mjue_code)
    ctx.edit_cell("SFPL", code=sfpl_code)
print("Updated MJUe and SFPL!")
