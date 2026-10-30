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
        noise = rms_normalize(batch["waveform"].to(device), 0.07925)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            s_logits = student(noise).logits
            t_logits = teacher(noise).logits
        s_logits_list.append(s_logits.float().cpu().numpy())
        t_logits_list.append(t_logits.float().cpu().numpy())
        labels.append(batch["target"].argmax(dim=-1).numpy())
        snrs.append(batch["snr_db"].numpy())

s_logits = np.concatenate(s_logits_list, axis=0)
t_logits = np.concatenate(t_logits_list, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

print("Logit Blending:")
for w in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]:
    blended_l = w * s_logits + (1 - w) * t_logits
    probs = torch.from_numpy(blended_l).softmax(dim=-1).numpy()
    res = snr_scores(probs, labels, snrs, 36)
    print(f"  w_student={w:.2f}: top1={res['top1_accuracy']:.4f}, macro_acc={res['macro_acc']:.4f}, mAP={res['mAP']:.4f}, mAP_macro={res['mAP_macro']:.4f}")

# Temperature scaled
print("\nTemperature Scaled (T_student=1.2, T_teacher=1.0):")
for w in [0.25, 0.3, 0.35, 0.4]:
    s_p = torch.from_numpy(s_logits / 1.2).softmax(dim=-1).numpy()
    t_p = torch.from_numpy(t_logits / 1.0).softmax(dim=-1).numpy()
    probs = w * s_p + (1 - w) * t_p
    res = snr_scores(probs, labels, snrs, 36)
    print(f"  w={w:.2f}: top1={res['top1_accuracy']:.4f}, macro_acc={res['macro_acc']:.4f}, mAP={res['mAP']:.4f}, mAP_macro={res['mAP_macro']:.4f}")
"""

res = c.run(code)
print(res)
