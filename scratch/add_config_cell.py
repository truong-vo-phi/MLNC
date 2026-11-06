import marimo._code_mode as cm

config_code = '''import torch
from pathlib import Path

# ==============================================================================
# 1. PARAMETERS & RUNTIME CONFIGURATION
# Set RUN_TRAINING = True for full 15-epoch training; False for quick smoke-test
# ==============================================================================
RUN_TRAINING = False

# Paths
DATA_ROOT = Path("/datasets/mix-dataset-v2")
FRCRN_CACHE_ROOT = Path("/datasets/mix-frcrn")
RESULTS_DIR = Path("/results/dual-ced-tiny")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset properties
NUM_CLASSES = 36
SAMPLE_RATE = 16000
DURATION = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 64,000 samples
SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

# Training Hyperparameters (mix_dataset_v2 & review-3 standards)
EPOCHS = 15 if RUN_TRAINING else 1
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 2  # Effective batch size = 64
BACKBONE_LR = 3.0e-5
HEAD_LR = 3.0e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 2
ROUTER_THRESHOLD_DB = 18.0

# Device & Mixed Precision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16

print(f"[Config] RUN_TRAINING: {RUN_TRAINING} (Epochs={EPOCHS}, Batch={BATCH_SIZE}x{GRADIENT_ACCUMULATION}=64)")
print(f"[Config] Device: {DEVICE} | AMP bf16: {AMP_ENABLED} | Results: {RESULTS_DIR}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(code=config_code, after="nGLA", hide_code=False, name="config_params")
    ctx.run_cell(cid)
print("Added config_params cell successfully")
