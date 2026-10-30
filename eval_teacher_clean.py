from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=120.0
)

code = r"""
import sys, os
os.chdir("/marimo/imp/kd-v5")
for k in list(sys.modules.keys()):
    if k.startswith(('src', 'scripts')):
        del sys.modules[k]
sys.path.insert(0, "/marimo/imp/kd-v5")

import torch
import numpy as np
from pathlib import Path
from src.metrics import snr_scores
from src.utils.files import rms_normalize
from src.model.beat.checkpoint import load_fold_teacher
from src.train.config import DatasetConfig
from scripts.load_dataset import build_cached_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_path = "/marimo/pretrained/beats_teacher_fold0_weights.pt"
teacher, meta = load_fold_teacher(teacher_path, "/marimo/imp/kd-v5/config/beat.yaml", device)
print("Teacher meta:", meta)

d_cfg = DatasetConfig(name="mixture-36l", duration=4.0)
cache_dir = Path("/marimo/dataset/cache")
val_loader = build_cached_loader(d_cfg, "validation", cache_dir, 64, None, 0, shuffle=False)

print(f"Evaluating teacher on {len(val_loader.dataset)} validation samples...")
probs, labels, snrs = [], [], []
with torch.no_grad():
    for batch in val_loader:
        noise = rms_normalize(batch["waveform"].to(device), 0.07925)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = teacher(noise).logits
        probs.append(logits.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch["target"].argmax(dim=-1).numpy())
        snrs.append(batch["snr_db"].numpy())

probs = np.concatenate(probs, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)
scores = snr_scores(probs, labels, snrs, 36)
print("Teacher results:")
for k in ["top1_accuracy", "macro_acc", "mAP", "mAP_macro", "macro_auc"]:
    print(f"  {k}: {scores[k]:.4f}")
for snr in sorted(np.unique(snrs)):
    print(f"  SNR {int(snr):3d} dB: acc={scores[f'acc_snr_{int(snr)}']:.4f}, mAP={scores[f'mAP_snr_{int(snr)}']:.4f}")
"""

res = c.run(code)
print(res)
