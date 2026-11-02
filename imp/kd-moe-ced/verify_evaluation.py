import os
import sys
from pathlib import Path

# Add kd-moe-ced to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import marimo_client

code = r"""
import torch, numpy as np
import torch.nn.functional as F
from pathlib import Path
from src.metrics import compute_snr_breakdown
from src.models.split_moe import CEDSingleMoESplitFFN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running verification on:', device)

# Load student
model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=False).to(device)
ckpt_path = '/marimo/checkpoints/best_champion_moe.pt'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt)
model.eval()

# Load eval data
val_waveforms = np.load('/marimo/datasets/validation.npy', mmap_mode='r')
import pandas as pd, json
df_manifest = pd.read_csv('/marimo/datasets/mix-dataset-v2/eval/manifest.csv')
val_targets = np.argmax(np.array([json.loads(x) for x in df_manifest['multi_hot_36']]), axis=-1)
val_snrs = df_manifest['target_snr_db'].to_numpy()

# 1. Evaluate Student Standalone with 3-Way TTA
all_probs, all_alphas = [], []
with torch.no_grad():
    for i in range(0, len(val_waveforms), 128):
        wave = torch.from_numpy(np.array(val_waveforms[i:i+128])).to(device)
        def fwd(shift):
            if shift > 0:
                w = F.pad(wave[:, shift:], (0, shift), mode='reflect')
            elif shift < 0:
                s = abs(shift)
                w = F.pad(wave[:, :-s], (s, 0), mode='reflect')
            else:
                w = wave
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(w, return_dict=True)
            return out['logits'].float(), out['alpha'].squeeze(-1).float()
        l0, a0 = fwd(0)
        lp, _ = fwd(800)
        ln, _ = fwd(-800)
        lf = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
        p = F.softmax(lf / 1.35, dim=-1).cpu().numpy()
        all_probs.append(p)
        all_alphas.append(a0.cpu().numpy())

p_student = np.concatenate(all_probs, axis=0)
alphas = np.concatenate(all_alphas, axis=0)

res_student = compute_snr_breakdown(p_student, val_targets, val_snrs, alphas=alphas)
print(f"1. Standalone Student CED-SingleMoE (6.62M): Top-1 = {res_student['all']['top1']*100:.2f}%, Top-3 = {res_student['all']['top3']*100:.2f}%, mAP = {res_student['all']['map']*100:.2f}%")

# 2. Evaluate Champion Pipeline v2 Ensemble with Dual Teachers
val_t0 = np.load('/marimo/datasets/teacher_cache/val_t0_logits.npy')
val_tema = np.load('/marimo/datasets/teacher_cache/val_tema_logits.npy')
p_t0 = F.softmax(torch.from_numpy(val_t0) / 1.20, dim=-1).numpy()
p_tema = F.softmax(torch.from_numpy(val_tema) / 1.20, dim=-1).numpy()

alpha_col = alphas.reshape(-1, 1)
w_t0 = (1.0 - alpha_col) * 0.45 + alpha_col * 0.30
w_tema = (1.0 - alpha_col) * 0.30 + alpha_col * 0.45

p_final = 0.25 * p_student + w_t0 * p_t0 + w_tema * p_tema
res_champion = compute_snr_breakdown(p_final, val_targets, val_snrs, alphas=alphas)
print(f"2. Champion Pipeline v2 Ensemble (187.3M):  Top-1 = {res_champion['all']['top1']*100:.2f}%, Top-3 = {res_champion['all']['top3']*100:.2f}%, mAP = {res_champion['all']['map']*100:.2f}%")
print(f"   At SNR -5dB: Top-1 = {res_champion['-5dB']['top1']*100:.2f}%")
print(f"   At SNR 0dB:  Top-1 = {res_champion['0dB']['top1']*100:.2f}%")
print(f"   At SNR 5dB:  Top-1 = {res_champion['5dB']['top1']*100:.2f}%")
print(f"   At SNR 10dB: Top-1 = {res_champion['10dB']['top1']*100:.2f}%")
print(f"   At SNR 15dB: Top-1 = {res_champion['15dB']['top1']*100:.2f}%")
print(f"   At SNR 20dB: Top-1 = {res_champion['20dB']['top1']*100:.2f}%")
"""

marimo_client.execute_remote(code)
