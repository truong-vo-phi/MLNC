from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=120.0
)

code = r"""
import sys
for k in list(sys.modules.keys()):
    if k == 'src' or k.startswith('src.'):
        del sys.modules[k]
sys.path.insert(0, "/marimo/imp/kd-v5")
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.metrics import snr_scores
from src.utils.files import rms_normalize
from src.model.beat import BEATClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_path = "/marimo/pretrained/beats_teacher_fold0_weights.pt"
from src.model.beat.checkpoint import load_fold_teacher
teacher, meta = load_fold_teacher(teacher_path, "/marimo/imp/kd-v5/config/beat.yaml", device)
print("Teacher meta:", meta)

val_npy = np.load("/marimo/dataset/cache/validation.npy", mmap_mode="r")
manifest = pd.read_csv("/marimo/dataset/mix-dataset/mix-dataset-v2/eval/manifest.csv")
print("Manifest columns:", list(manifest.columns))
print("Head:\n", manifest.head(2))
probs = []
bs = 64
with torch.no_grad():
    for i in range(0, len(val_npy), bs):
        batch = torch.from_numpy(np.array(val_npy[i:i+bs])).to(device)
        noise = rms_normalize(batch, 0.07925)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = teacher(noise).logits
        probs.append(logits.float().softmax(dim=-1).cpu().numpy())

probs = np.concatenate(probs, axis=0)
scores = snr_scores(probs, labels, snrs, 36)
print("Teacher results:")
for k in ["top1_accuracy", "macro_acc", "mAP", "mAP_macro", "macro_auc"]:
    print(f"  {k}: {scores[k]:.4f}")
for snr in sorted(np.unique(snrs)):
    print(f"  SNR {snr:3d} dB: acc={scores[f'acc_snr_{int(snr)}']:.4f}, mAP={scores[f'mAP_snr_{int(snr)}']:.4f}")
"""

res = c.run(code)
print(res)
