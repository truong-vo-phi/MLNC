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
from main import build_student, read_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw = read_config(Path("config/kd_ced_tiny.yaml"))

# Load student
student = build_student(raw).to(device)
ckpt_path = Path("/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-17-51-19/student/last_best_checkpoint.pt")
student.load_pretrained(ckpt_path)
student.eval()

# Load teacher
teacher, meta = load_fold_teacher(Path("/marimo/pretrained/beats_teacher_fold0_weights.pt"), Path("config/beat.yaml"), device)
teacher.eval()

d_cfg = DatasetConfig(name="mixture-36l", duration=4.0)
cache_dir = Path("/marimo/dataset/cache")
val_loader = build_cached_loader(d_cfg, "validation", cache_dir, 64, None, 0, shuffle=False)

s_probs, t_probs, labels, snrs = [], [], [], []
with torch.no_grad():
    for batch in val_loader:
        noise = rms_normalize(batch["waveform"].to(device), 0.07925)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            s_logits = student(noise).logits
            t_logits = teacher(noise).logits
        s_probs.append(s_logits.float().softmax(dim=-1).cpu().numpy())
        t_probs.append(t_logits.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch["target"].argmax(dim=-1).numpy())
        snrs.append(batch["snr_db"].numpy())

s_probs = np.concatenate(s_probs, axis=0)
t_probs = np.concatenate(t_probs, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

s_scores = snr_scores(s_probs, labels, snrs, 36)
t_scores = snr_scores(t_probs, labels, snrs, 36)

print("Student alone (Epoch 3):")
print(f"  top1: {s_scores['top1_accuracy']:.4f}, macro_acc: {s_scores['macro_acc']:.4f}, mAP: {s_scores['mAP']:.4f}, mAP_macro: {s_scores['mAP_macro']:.4f}")

print("Teacher alone:")
print(f"  top1: {t_scores['top1_accuracy']:.4f}, macro_acc: {t_scores['macro_acc']:.4f}, mAP: {t_scores['mAP']:.4f}, mAP_macro: {t_scores['mAP_macro']:.4f}")

for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ens_probs = w * s_probs + (1 - w) * t_probs
    e_scores = snr_scores(ens_probs, labels, snrs, 36)
    print(f"Ensemble (w_student={w:.1f}): top1={e_scores['top1_accuracy']:.4f}, macro_acc={e_scores['macro_acc']:.4f}, mAP={e_scores['mAP']:.4f}, mAP_macro={e_scores['mAP_macro']:.4f}")
"""

res = c.run(code)
print(res)
