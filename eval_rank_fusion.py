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
from scipy.stats import rankdata
from src.metrics import snr_scores
from src.utils.files import rms_normalize
from src.model.beat.checkpoint import load_fold_teacher
from src.train.config import DatasetConfig
from scripts.load_dataset import build_cached_loader
from main import build_student, read_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw = read_config(Path("config/kd_ced_tiny.yaml"))

student = build_student(raw).to(device)
ckpt_path = Path("/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt")
student.load_pretrained(ckpt_path)
student.eval()

teacher, meta = load_fold_teacher(Path("/marimo/pretrained/beats_teacher_fold0_weights.pt"), Path("config/beat.yaml"), device)
teacher.eval()

d_cfg = DatasetConfig(name="mixture-36l", duration=4.0)
cache_dir = Path("/marimo/dataset/cache")
val_loader = build_cached_loader(d_cfg, "validation", cache_dir, 64, None, 0, shuffle=False)

s_probs_list, t_probs_list, s_logits_list, t_logits_list, labels, snrs = [], [], [], [], [], []
with torch.no_grad():
    for batch in val_loader:
        noise = rms_normalize(batch["waveform"].to(device), 0.07925)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            s_l = student(noise).logits
            t_l = teacher(noise).logits
        s_logits_list.append(s_l.float().cpu().numpy())
        t_logits_list.append(t_l.float().cpu().numpy())
        s_probs_list.append(s_l.float().softmax(dim=-1).cpu().numpy())
        t_probs_list.append(t_l.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch["target"].argmax(dim=-1).numpy())
        snrs.append(batch["snr_db"].numpy())

s_logits = np.concatenate(s_logits_list, axis=0)
t_logits = np.concatenate(t_logits_list, axis=0)
s_probs = np.concatenate(s_probs_list, axis=0)
t_probs = np.concatenate(t_probs_list, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

# 1. Geometric Mean: sqrt(s_prob * t_prob)
for w in [0.3, 0.4, 0.5]:
    geom = (s_probs ** w) * (t_probs ** (1 - w))
    res = snr_scores(geom, labels, snrs, 36)
    print(f"Geometric Mean (w={w:.2f}): top1={res['top1_accuracy']:.4f}, macro_acc={res['macro_acc']:.4f}, mAP={res['mAP']:.4f}, mAP_macro={res['mAP_macro']:.4f}")

# 2. Normalized Rank Averaging per class
s_ranks = np.zeros_like(s_probs)
t_ranks = np.zeros_like(t_probs)
for c in range(36):
    s_ranks[:, c] = rankdata(s_probs[:, c]) / len(s_probs)
    t_ranks[:, c] = rankdata(t_probs[:, c]) / len(t_probs)

for w in [0.3, 0.35, 0.4, 0.45, 0.5]:
    blended_ranks = w * s_ranks + (1 - w) * t_ranks
    res = snr_scores(blended_ranks, labels, snrs, 36)
    print(f"Rank Averaging (w={w:.2f}): top1={res['top1_accuracy']:.4f}, macro_acc={res['macro_acc']:.4f}, mAP={res['mAP']:.4f}, mAP_macro={res['mAP_macro']:.4f}")

# 3. Softmax temperature on logits before adding
for temp in [0.7, 0.8, 0.9, 1.0, 1.2]:
    p = (0.4 * s_logits + 0.6 * t_logits) / temp
    res = snr_scores(torch.from_numpy(p).softmax(dim=-1).numpy(), labels, snrs, 36)
    print(f"Logit Blend Temp {temp:.1f}: top1={res['top1_accuracy']:.4f}, macro_acc={res['macro_acc']:.4f}, mAP={res['mAP']:.4f}, mAP_macro={res['mAP_macro']:.4f}")
"""

res = c.run(code)
print(res)
