import sys
from pathlib import Path

# Add kd-moe-ced to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import marimo_client

code = r"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.metrics import compute_snr_breakdown

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running Diagnostic D2 & D3 on:', device)

# Load manifest and true labels
manifest_path = Path('/marimo/datasets/mix-dataset-v2/eval/manifest.csv')
df = pd.read_csv(manifest_path)
targets = np.argmax(np.array([json.loads(x) for x in df['multi_hot_36']]), axis=-1)
snrs = df['target_snr_db'].to_numpy()

# Load teacher logits
t0_raw = np.load('/marimo/datasets/teacher_cache/val_t0_logits.npy')
tema_raw = np.load('/marimo/datasets/teacher_cache/val_tema_logits.npy')

# Compute probabilities (T=1.20)
p_t0 = F.softmax(torch.from_numpy(t0_raw) / 1.20, dim=-1).numpy()
p_tema = F.softmax(torch.from_numpy(tema_raw) / 1.20, dim=-1).numpy()

# Evaluate Fold0 standalone
res_t0 = compute_snr_breakdown(p_t0, targets, snrs)
# Evaluate EMA standalone
res_tema = compute_snr_breakdown(p_tema, targets, snrs)

# Evaluate simple 50-50 Ensemble of Teachers
p_ensemble = 0.5 * p_t0 + 0.5 * p_tema
res_ens = compute_snr_breakdown(p_ensemble, targets, snrs)

# Also load Student predictions from best_champion_moe
from src.models.split_moe import CEDSingleMoESplitFFN
model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=False).to(device)
ckpt = torch.load('/marimo/checkpoints/best_champion_moe.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt)
model.eval()

val_waveforms = np.load('/marimo/datasets/validation.npy', mmap_mode='r')
all_probs = []
with torch.no_grad():
    for i in range(0, len(val_waveforms), 128):
        wave = torch.from_numpy(np.array(val_waveforms[i:i+128])).to(device)
        def fwd(shift):
            if shift > 0: w = F.pad(wave[:, shift:], (0, shift), mode='reflect')
            elif shift < 0: s = abs(shift); w = F.pad(wave[:, :-s], (s, 0), mode='reflect')
            else: w = wave
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(w, return_dict=True)
            return out['logits'].float()
        l0 = fwd(0)
        lp = fwd(800)
        ln = fwd(-800)
        lf = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
        p = F.softmax(lf / 1.35, dim=-1).cpu().numpy()
        all_probs.append(p)

p_student = np.concatenate(all_probs, axis=0)
res_student = compute_snr_breakdown(p_student, targets, snrs)

# Pred labels
pred_t0 = np.argmax(p_t0, axis=-1)
pred_tema = np.argmax(p_tema, axis=-1)
pred_student = np.argmax(p_student, axis=-1)

# Teacher Agreement: % where Fold0 and EMA predict the exact same class
# Teacher-Student Agreement
snr_levels = [-5, 0, 5, 10, 15, 20]
print("\n" + "="*80)
print("DIAGNOSTIC TABLE D2: PERFORMANCE BREAKDOWN PER SNR")
print("="*80)
print(f"{'SNR':<6} | {'Fold0 Top-1':<12} | {'EMA Top-1':<12} | {'Ensemble T':<12} | {'Student (Ours)':<14} | {'Winner':<8}")
print("-" * 72)

for snr in snr_levels:
    snr_k = f"{snr}dB"
    acc_t0 = res_t0[snr_k]['top1'] * 100
    acc_tema = res_tema[snr_k]['top1'] * 100
    acc_ens = res_ens[snr_k]['top1'] * 100
    acc_s = res_student[snr_k]['top1'] * 100
    winner = "Fold0" if acc_t0 > acc_tema else "EMA"
    print(f"{snr_k:<6} | {acc_t0:9.2f}%   | {acc_tema:9.2f}%   | {acc_ens:9.2f}%   | {acc_s:11.2f}%   | {winner:<8}")

print("-" * 72)
print(f"{'ALL':<6} | {res_t0['all']['top1']*100:9.2f}%   | {res_tema['all']['top1']*100:9.2f}%   | {res_ens['all']['top1']*100:9.2f}%   | {res_student['all']['top1']*100:11.2f}%   |")
print("="*80)

print("\n" + "="*80)
print("DIAGNOSTIC TABLE D3: TEACHER & STUDENT AGREEMENT ANALYSIS")
print("="*80)
print(f"{'SNR':<6} | {'Fold0-EMA Agree':<16} | {'Both Correct':<14} | {'Both Wrong':<12} | {'Only Fold0 Correct':<19} | {'Only EMA Correct':<18}")
print("-" * 90)

for snr in snr_levels:
    mask = (snrs == snr)
    n = mask.sum()
    agree = (pred_t0[mask] == pred_tema[mask]).mean() * 100
    c_t0 = (pred_t0[mask] == targets[mask])
    c_tema = (pred_tema[mask] == targets[mask])
    both_c = (c_t0 & c_tema).mean() * 100
    both_w = (~c_t0 & ~c_tema).mean() * 100
    only_t0 = (c_t0 & ~c_tema).mean() * 100
    only_tema = (~c_t0 & c_tema).mean() * 100
    print(f"{snr}dB   | {agree:13.2f}%   | {both_c:11.2f}%   | {both_w:9.2f}%   | {only_t0:16.2f}%    | {only_tema:15.2f}%")

print("-" * 90)
all_agree = (pred_t0 == pred_tema).mean() * 100
all_c_t0 = (pred_t0 == targets)
all_c_tema = (pred_tema == targets)
all_both_c = (all_c_t0 & all_c_tema).mean() * 100
all_both_w = (~all_c_t0 & ~all_c_tema).mean() * 100
all_only_t0 = (all_c_t0 & ~all_c_tema).mean() * 100
all_only_tema = (~all_c_t0 & all_c_tema).mean() * 100
print(f"{'ALL':<6} | {all_agree:13.2f}%   | {all_both_c:11.2f}%   | {all_both_w:9.2f}%   | {all_only_t0:16.2f}%    | {all_only_tema:15.2f}%")
print("="*80)

# Student Agreement with Teachers
print("\nSTUDENT-TEACHER AGREEMENT:")
for snr in snr_levels:
    mask = (snrs == snr)
    s_agree_t0 = (pred_student[mask] == pred_t0[mask]).mean() * 100
    s_agree_tema = (pred_student[mask] == pred_tema[mask]).mean() * 100
    print(f"  {snr}dB: Student matches Fold0 = {s_agree_t0:5.2f}% | Student matches EMA = {s_agree_tema:5.2f}%")
"""

marimo_client.execute_remote(code)
