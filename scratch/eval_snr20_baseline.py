from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=180.0,
)

code = """
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from main import build_student
from src.model.beat import load_fold_teacher
from src.metrics import snr_scores
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.utils.files import rms_normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Student
with open('/marimo/imp/kd-v5/config/kd_ced_tiny.yaml') as f:
    cfg = yaml.safe_load(f)

student = build_student(cfg)
ckpt_path = '/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt'
student.load_pretrained(ckpt_path)
student.to(device).eval()

# Load Teacher
teacher, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_fold0_weights.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
teacher.eval()

# Load Teacher EMA
teacher_ema, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_ema_standalone.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
teacher_ema.eval()

base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

s_probs, t_probs, tema_probs, labels, snrs = [], [], [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc='Inferencing all models'):
        wave = batch['waveform'].to(device, non_blocking=True)
        noise = rms_normalize(wave, 0.07925)
        
        with torch.autocast(device.type, dtype=torch.bfloat16):
            s_logits = student(noise).logits
            t_logits = teacher(noise).logits
            tema_logits = teacher_ema(noise).logits
            
        s_probs.append(s_logits.float().softmax(dim=-1).cpu().numpy())
        t_probs.append(t_logits.float().softmax(dim=-1).cpu().numpy())
        tema_probs.append(tema_logits.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch['target'].argmax(dim=-1).numpy())
        snrs.append(batch['snr_db'].numpy())

s_probs = np.concatenate(s_probs, axis=0)
t_probs = np.concatenate(t_probs, axis=0)
tema_probs = np.concatenate(tema_probs, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

mask_20 = (snrs == 20.0)
mask_other = (snrs != 20.0)

print('=============================================================================')
print('ĐỐI CHIẾU: PHÂN TÁCH SNR 20 (509 clips) VÀ CÁC MỨC SNR CÒN LẠI (2555 clips)')
print('=============================================================================')
models = [
    ('Student CED-Tiny (15ep)', s_probs),
    ('Teacher BEATs Fold0', t_probs),
    ('Teacher BEATs EMA', tema_probs),
    ('Ensemble (0.4 S + 0.6 T_fold0)', 0.4 * s_probs + 0.6 * t_probs),
    ('Ensemble (0.4 S + 0.6 T_ema)', 0.4 * s_probs + 0.6 * tema_probs),
    ('Ensemble (0.3 S + 0.4 T_fold0 + 0.3 T_ema)', 0.3 * s_probs + 0.4 * t_probs + 0.3 * tema_probs)
]

for name, p in models:
    acc_all = (p.argmax(axis=-1) == labels).mean() * 100
    acc_other = (p[mask_other].argmax(axis=-1) == labels[mask_other]).mean() * 100
    acc_20 = (p[mask_20].argmax(axis=-1) == labels[mask_20]).mean() * 100
    print(f'{name:<42} | OVERALL: {acc_all:6.2f}% | Rest (-5..15): {acc_other:6.2f}% | SNR 20: {acc_20:6.2f}%')
print('=============================================================================')
"""

scratch_file = "scratch_inspect_snr20.py"
run_code = f"""
import subprocess
from pathlib import Path
Path('/marimo/imp/kd-v5/{scratch_file}').write_text({repr(code)}, encoding='utf-8')
res = subprocess.run(['python', '/marimo/imp/kd-v5/{scratch_file}'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')
print(res.stdout)
print(res.stderr)
"""

out = client.run(run_code)
sys.stdout.buffer.write(out.encode("utf-8"))
