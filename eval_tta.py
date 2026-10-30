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
import torch.nn.functional as F
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

student = build_student(raw).to(device)
ckpt_path = Path("/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt")
student.load_pretrained(ckpt_path)
student.eval()

teacher, meta = load_fold_teacher(Path("/marimo/pretrained/beats_teacher_fold0_weights.pt"), Path("config/beat.yaml"), device)
teacher.eval()

d_cfg = DatasetConfig(name="mixture-36l", duration=4.0)
cache_dir = Path("/marimo/dataset/cache")
val_loader = build_cached_loader(d_cfg, "validation", cache_dir, 64, None, 0, shuffle=False)

s_logits_list, t_logits_list, labels, snrs = [], [], [], []
with torch.no_grad():
    for batch in val_loader:
        wave = batch["waveform"].to(device)
        noise = rms_normalize(wave, 0.07925)
        
        # Student TTA: 3 crops (offset 0, 1, 2)
        mel = student.frontend.logmel(noise)
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std = mel.std(dim=(-2, -1), keepdim=True)
        mel_norm = (mel - mean) / (std + 1e-5)
        
        crops = [mel_norm[:, :, i:i+398].unsqueeze(1) for i in [0, 1]]
        with torch.autocast(device.type, dtype=torch.bfloat16):
            s_preds = [student.vit(c) for c in crops]
            s_l = torch.stack(s_preds).mean(dim=0)
            
            # Teacher TTA: original, shift +80, shift -80
            t_orig = teacher(noise).logits
            t_r = teacher(torch.roll(noise, 80, dims=-1)).logits
            t_l = teacher(torch.roll(noise, -80, dims=-1)).logits
            t_l_mean = (t_orig + t_r + t_l) / 3.0

        s_logits_list.append(s_l.float().cpu().numpy())
        t_logits_list.append(t_l_mean.float().cpu().numpy())
        labels.append(batch["target"].argmax(dim=-1).numpy())
        snrs.append(batch["snr_db"].numpy())

s_logits = np.concatenate(s_logits_list, axis=0)
t_logits = np.concatenate(t_logits_list, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

# Student TTA alone
s_res = snr_scores(torch.from_numpy(s_logits).softmax(dim=-1).numpy(), labels, snrs, 36)
print("Student TTA alone:")
print(f"  top1: {s_res['top1_accuracy']:.4f}, mAP: {s_res['mAP']:.4f}, mAP_macro: {s_res['mAP_macro']:.4f}")

# Teacher TTA alone
t_res = snr_scores(torch.from_numpy(t_logits).softmax(dim=-1).numpy(), labels, snrs, 36)
print("Teacher TTA alone:")
print(f"  top1: {t_res['top1_accuracy']:.4f}, mAP: {t_res['mAP']:.4f}, mAP_macro: {t_res['mAP_macro']:.4f}")

print("\nEnsemble with TTA:")
for w in [0.3, 0.35, 0.38, 0.4, 0.42, 0.45]:
    blended = w * s_logits + (1 - w) * t_logits
    res = snr_scores(torch.from_numpy(blended).softmax(dim=-1).numpy(), labels, snrs, 36)
    print(f"  w={w:.2f}: top1={res['top1_accuracy']:.4f}, macro_acc={res['macro_acc']:.4f}, mAP={res['mAP']:.4f}, mAP_macro={res['mAP_macro']:.4f}")
"""

res = c.run(code)
print(res)
