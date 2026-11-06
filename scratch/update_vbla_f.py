import marimo._code_mode as cm

vbla_code = """import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from src.dataset import MixtureDataset
from src.models.ced_moe_split import CEDSingleMoESplitFFN
from src.metrics import compute_9_metrics_dict, compute_snr_breakdown

# 1. Load config from imp/review-3/config/config.yaml
_config_path = Path("/marimo/config/config.yaml")
with open(_config_path, "r", encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

DATA_ROOT = "/datasets/mix-dataset-v2" if Path("/datasets/mix-dataset-v2").is_dir() else cfg["dataset"]["data_root"]
PRETRAINED_PATH = "/marimo/pretrained/model.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Build full datasets
ds_train = MixtureDataset(DATA_ROOT, split="train")
ds_eval = MixtureDataset(DATA_ROOT, split="eval")

# 3. Initialize CED-SingleMoE-SplitFFN (~6.62M params)
moe_model = CEDSingleMoESplitFFN(
    num_classes=cfg["dataset"]["num_classes"],
    load_pretrained=True,
    pretrained_path=PRETRAINED_PATH,
).to(DEVICE)

_p = moe_model.get_num_params()

mo.md(
    f\"\"\"
### \\U0001f2699\\U0000fe0f [1] Cau Hinh & Khoi Tao CED-SplitFFN-MoE
- **Thiet bi:** `{DEVICE}` ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})
- **Tap du lieu:** `{DATA_ROOT}` (Train: `{len(ds_train):,}` mau, Eval: `{len(ds_eval):,}` mau)
- **Tong so tham so:** `{_p['total']:,}` (Backbone + SplitFFN: `{_p['trainable']:,}`, Router: `{_p['router']:,}`)
- **AudioSet CED-Tiny Pretrained Weights:** `{moe_model.pretrained_loaded_tensors}` tensors da nap.
- **Muc SNR danh gia:** `[-5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB]`
\"\"\"
)
"""

async def update_vbla():
    async with cm.get_context() as ctx:
        ctx.edit_cell("vblA", code=vbla_code, hide_code=False)
        ctx.run_cell("vblA")

await update_vbla()
print("vblA updated!")
