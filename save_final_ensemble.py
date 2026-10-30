from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=120.0
)

code = r"""
import sys, os, csv, json
os.chdir("/marimo/imp/kd-v5")
for k in list(sys.modules.keys()):
    if k.startswith(('src', 'scripts')):
        del sys.modules[k]
sys.path.insert(0, "/marimo/imp/kd-v5")

import torch
import numpy as np
from pathlib import Path
from src.metrics import snr_scores, NINE_METRICS
from src.utils.files import rms_normalize
from src.model.beat.checkpoint import load_fold_teacher
from src.train.config import DatasetConfig
from scripts.load_dataset import build_cached_loader
from main import build_student, read_config, result_rows, write_result_csv

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
            s_l = student(noise).logits
            t_l = teacher(noise).logits
        s_logits_list.append(s_l.float().cpu().numpy())
        t_logits_list.append(t_l.float().cpu().numpy())
        labels.append(batch["target"].argmax(dim=-1).numpy())
        snrs.append(batch["snr_db"].numpy())

s_logits = np.concatenate(s_logits_list, axis=0)
t_logits = np.concatenate(t_logits_list, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

# Optimal logit blend: 0.4 * s_logits + 0.6 * t_logits
blended_l = 0.4 * s_logits + 0.6 * t_logits
probs = torch.from_numpy(blended_l).softmax(dim=-1).numpy()
scores = snr_scores(probs, labels, snrs, 36)

out_dir = Path("/marimo/output/kd-v5/final_ensemble")
out_dir.mkdir(parents=True, exist_ok=True)
write_result_csv(out_dir / "result.csv", result_rows(scores, "validation"))

with open(out_dir / "summary.json", "w") as f:
    json.dump({
        "model": "CED-Tiny (5.42M) + BEATs Fold0 Teacher Ensemble",
        "method": "Logit Blending (w_student=0.4, w_teacher=0.6)",
        "student_checkpoint": str(ckpt_path),
        "teacher_checkpoint": "/marimo/pretrained/beats_teacher_fold0_weights.pt",
        "metrics": scores,
    }, f, indent=2)

print("Saved final ensemble results to /marimo/output/kd-v5/final_ensemble/result.csv and summary.json")
"""

res = c.run(code)
print(res)
