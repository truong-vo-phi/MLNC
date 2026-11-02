import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

fix_cell3_code = '''
# 1. Cấu hình đường dẫn hệ thống và nạp các thư viện cốt lõi
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Đảm bảo in tiếng Việt chuẩn UTF-8 trên mọi môi trường
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tự động định vị thư mục gốc gói kd-moe-ced (nơi chứa src/frontend.py)
PKG_ROOT = None
search_candidates = [
    Path.cwd() / "imp" / "kd-moe-ced",
    Path.cwd(),
    Path("/marimo/imp/kd-moe-ced"),
    Path("d:/PROJECT/MLNC/imp/kd-moe-ced"),
]
for candidate in search_candidates:
    if (candidate / "src" / "frontend.py").is_file():
        PKG_ROOT = candidate
        break

if PKG_ROOT is None:
    PKG_ROOT = Path("/marimo/imp/kd-moe-ced")

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if "/marimo" not in sys.path:
    sys.path.insert(0, "/marimo")

# Nạp các module tự phát triển từ package src
from src.frontend import TARGET_RMS, AcousticFeatureExtractor, LogMel, SpecAugment, rms_normalize
from src.router import SelfContainedSpectralTokenRouter
from src.models.vit_grep import ViTSBias, BiasAttention, SharedBias
from src.models.split_moe import CEDSingleMoESplitFFN, SplitMoEFFN
from src.metrics import compute_topk_acc, compute_map, compute_auc, compute_snr_breakdown, compute_9_metrics_dict
from src.pipeline import ChampionPipeline

# Thiết lập thiết bị tính toán tối ưu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Thư mục dự án: {PKG_ROOT}")
print(f"-> Thiết bị tính toán được chọn: {device}")
torch.manual_seed(42)
np.random.seed(42)
'''

update_script = f"""
import marimo._code_mode as cm

async def fix():
    async with cm.get_context() as ctx:
        ctx.edit_cell("vblA", {repr(fix_cell3_code.strip())})
        ctx.run_cell("vblA")

await fix()
"""

rc = execute_remote(update_script)
print(f"Cell 3 update executed with code {rc}")
