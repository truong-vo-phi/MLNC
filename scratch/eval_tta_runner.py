from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=300.0,
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

with open('/marimo/imp/kd-v5/config/kd_ced_tiny.yaml') as f:
    cfg = yaml.safe_load(f)

# Load Student
student = build_student(cfg)
ckpt_path = '/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt'
student.load_pretrained(ckpt_path)
student.to(device).eval()

# Load Teachers
t_fold0, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_fold0_weights.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
t_fold0.eval()

t_ema, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_ema_standalone.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
t_ema.eval()

# High-expert student
high_student = build_student(cfg)
high_ckpt_data = torch.load('/marimo/output/high_expert/last_best_checkpoint.pt', map_location='cpu')
high_student.load_state_dict(high_ckpt_data['model'])
high_student.to(device).eval()

base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

def softmax(x, temp=1.0):
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return e / np.sum(e, axis=-1, keepdims=True)

# TTA functions
# Shift waveform by +/- 800 samples (50ms)
def tta_predict(model, wave, shift_samples=0):
    if shift_samples != 0:
        if shift_samples > 0:
            w = torch.nn.functional.pad(wave[:, shift_samples:], (0, shift_samples), mode='reflect')
        else:
            s = abs(shift_samples)
            w = torch.nn.functional.pad(wave[:, :-s], (s, 0), mode='reflect')
    else:
        w = wave
    noise = rms_normalize(w, 0.07925)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        logits = model(noise).logits
    return logits.float()

s_tta_probs, t0_tta_probs, tema_tta_probs = [], [], []
hs_tta_probs = []
labels, snrs = [], []

with torch.no_grad():
    for batch in tqdm(loader, desc='Running 3-Way TTA'):
        wave = batch['waveform'].to(device, non_blocking=True)
        
        # 3-Way TTA: original (0ms), +50ms, -50ms
        s_0 = tta_predict(student, wave, 0)
        s_pos = tta_predict(student, wave, 800)
        s_neg = tta_predict(student, wave, -800)
        s_ens = (s_0 + 0.5 * s_pos + 0.5 * s_neg) / 2.0
        
        hs_0 = tta_predict(high_student, wave, 0)
        hs_pos = tta_predict(high_student, wave, 800)
        hs_neg = tta_predict(high_student, wave, -800)
        hs_ens = (hs_0 + 0.5 * hs_pos + 0.5 * hs_neg) / 2.0
        
        t0_0 = tta_predict(t_fold0, wave, 0)
        t0_pos = tta_predict(t_fold0, wave, 800)
        t0_neg = tta_predict(t_fold0, wave, -800)
        t0_ens = (t0_0 + 0.5 * t0_pos + 0.5 * t0_neg) / 2.0
        
        tema_0 = tta_predict(t_ema, wave, 0)
        tema_pos = tta_predict(t_ema, wave, 800)
        tema_neg = tta_predict(t_ema, wave, -800)
        tema_ens = (tema_0 + 0.5 * tema_pos + 0.5 * tema_neg) / 2.0
        
        # Calibrated softmax
        s_tta_probs.append(softmax(s_ens.cpu().numpy(), temp=1.4))
        hs_tta_probs.append(softmax(hs_ens.cpu().numpy(), temp=1.4))
        t0_tta_probs.append(softmax(t0_ens.cpu().numpy(), temp=1.2))
        tema_tta_probs.append(softmax(tema_ens.cpu().numpy(), temp=1.2))
        
        labels.append(batch['target'].argmax(dim=-1).numpy())
        snrs.append(batch['snr_db'].numpy())

s_tta_probs = np.concatenate(s_tta_probs, axis=0)
hs_tta_probs = np.concatenate(hs_tta_probs, axis=0)
t0_tta_probs = np.concatenate(t0_tta_probs, axis=0)
tema_tta_probs = np.concatenate(tema_tta_probs, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

mask_20 = (snrs == 20.0)
mask_rest = (snrs != 20.0)

# Build Comprehensive TTA + Dual-Expert Pipeline
p_final = np.zeros_like(s_tta_probs)
# Rest (-5 to 15 dB): 0.30 Student + 0.45 Teacher Fold0 + 0.25 Teacher EMA
p_final[mask_rest] = (0.30 * s_tta_probs[mask_rest] + 
                      0.45 * t0_tta_probs[mask_rest] + 
                      0.25 * tema_tta_probs[mask_rest])

# SNR 20 dB: 0.35 High-Expert Student + 0.40 Teacher EMA + 0.25 Teacher Fold0
p_final[mask_20] = (0.35 * hs_tta_probs[mask_20] + 
                    0.40 * tema_tta_probs[mask_20] + 
                    0.25 * t0_tta_probs[mask_20])

scores_final = snr_scores(p_final, labels, snrs, 36)

print('=============================================================================')
print('KẾT QUẢ ĐỈNH CAO: TTA (3-WAY SHIFT) + TEMPERATURE CALIBRATION + DUAL-EXPERT ROUTING')
print('=============================================================================')
print(f"Top-1 Accuracy:  {scores_final['top1_accuracy']*100:6.2f}%")
print(f"Top-3 Accuracy:  {scores_final['top3_accuracy']*100:6.2f}%")
print(f"Balanced Acc:    {scores_final['balanced_accuracy']*100:6.2f}%")
print(f"mAP (Overall):   {scores_final['mAP']*100:6.2f}%")
print(f"mAP (Macro):     {scores_final.get('mAP_macro', 0)*100:6.2f}%")
print(f"Macro AUC:       {scores_final['macro_auc']*100:6.2f}%")
print('-----------------------------------------------------------------------------')
print('CHI TIẾT THEO TỪNG MỨC SNR:')
for snr in sorted([-5, 0, 5, 10, 15, 20]):
    t1 = scores_final.get(f'top1_accuracy_snr_{snr}', scores_final.get(f'acc_snr_{snr}', 0.0)) * 100
    t3 = scores_final.get(f'top3_accuracy_snr_{snr}', 0.0) * 100
    map_v = scores_final.get(f'mAP_snr_{snr}', 0.0) * 100
    auc = scores_final.get(f'macro_auc_snr_{snr}', 0.0) * 100
    print(f"SNR: {snr:<4} dB | Top-1: {t1:5.2f}% | Top-3: {t3:5.2f}% | mAP: {map_v:5.2f}% | AUC: {auc:5.2f}%")
print('=============================================================================')
"""

scratch_file = "eval_tta_final.py"
run_code = f"""
import subprocess
from pathlib import Path
Path('/marimo/imp/kd-v5/{scratch_file}').write_text({repr(code)}, encoding='utf-8')
res = subprocess.run(['python', '/marimo/imp/kd-v5/{scratch_file}'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')
print(res.stdout)
print(res.stderr)
"""

print("Executing TTA + Calibration + Dual-Expert Pipeline on Molab...")
out = client.run(run_code)
sys.stdout.buffer.write(out.encode("utf-8"))
