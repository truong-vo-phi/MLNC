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
from scipy.optimize import minimize

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

base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

s_logits_list, t0_logits_list, tema_logits_list = [], [], []
labels, snrs = [], []

with torch.no_grad():
    for batch in tqdm(loader, desc='Collecting uncalibrated logits'):
        wave = batch['waveform'].to(device, non_blocking=True)
        noise = rms_normalize(wave, 0.07925)
        
        with torch.autocast(device.type, dtype=torch.bfloat16):
            l_s = student(noise).logits
            l_t0 = t_fold0(noise).logits
            l_tema = t_ema(noise).logits
            
        s_logits_list.append(l_s.float().cpu().numpy())
        t0_logits_list.append(l_t0.float().cpu().numpy())
        tema_logits_list.append(l_tema.float().cpu().numpy())
        labels.append(batch['target'].argmax(dim=-1).numpy())
        snrs.append(batch['snr_db'].numpy())

s_logits = np.concatenate(s_logits_list, axis=0)
t0_logits = np.concatenate(t0_logits_list, axis=0)
tema_logits = np.concatenate(tema_logits_list, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

def softmax(x, temp=1.0):
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return e / np.sum(e, axis=-1, keepdims=True)

# Test 1: Grid Search over Temperature Calibration & Blending Weights
best_map = 0.0
best_params = None

print('--- Searching Optimal Temperature & Weight Calibration ---')
# Grid over weights and temperatures
for w_s in [0.2, 0.3, 0.35, 0.4, 0.45]:
    for w_t0 in [0.3, 0.4, 0.5, 0.6]:
        w_tema = 1.0 - w_s - w_t0
        if w_tema < -1e-5: continue
        w_tema = max(0.0, w_tema)
        
        for temp_s in [0.8, 1.0, 1.2, 1.5]:
            for temp_t in [0.8, 1.0, 1.2, 1.5]:
                p = (w_s * softmax(s_logits, temp_s) + 
                     w_t0 * softmax(t0_logits, temp_t) + 
                     w_tema * softmax(tema_logits, temp_t))
                
                # Check top-1 and mAP
                preds = p.argmax(axis=-1)
                acc = (preds == labels).mean()
                if acc >= 0.635:
                    scores = snr_scores(p, labels, snrs, 36)
                    cur_map = scores['mAP']
                    cur_map_macro = scores.get('mAP_macro', cur_map)
                    if cur_map > best_map:
                        best_map = cur_map
                        best_params = (w_s, w_t0, w_tema, temp_s, temp_t, acc, cur_map, cur_map_macro, scores['macro_auc'], scores['top3_accuracy'])

w_s, w_t0, w_tema, temp_s, temp_t, acc, map_v, map_macro, auc, top3 = best_params
print('=============================================================================')
print('KẾT QUẢ TỐI ƯU HÓA HIỆU CHUẨN NHIỆT ĐỘ (TEMPERATURE CALIBRATION & WEIGHTS):')
print(f'Weights: Student={w_s:.2f}, Teacher Fold0={w_t0:.2f}, Teacher EMA={w_tema:.2f}')
print(f'Temperatures: Temp_S={temp_s:.2f}, Temp_T={temp_t:.2f}')
print(f'OVERALL: Top-1: {acc*100:6.2f}% | Top-3: {top3*100:6.2f}% | mAP: {map_v*100:6.2f}% | mAP_macro: {map_macro*100:6.2f}% | AUC: {auc*100:6.2f}%')
print('=============================================================================')

# Test 2: Power-Rank Averaging (Non-linear Softmax Re-ranking)
print('\\n--- Testing Power-Rank Averaging ---')
ranks_s = np.argsort(np.argsort(s_logits, axis=-1), axis=-1) / 35.0
ranks_t0 = np.argsort(np.argsort(t0_logits, axis=-1), axis=-1) / 35.0
ranks_tema = np.argsort(np.argsort(tema_logits, axis=-1), axis=-1) / 35.0

for p_exp in [1.5, 2.0, 2.5, 3.0]:
    for ws in [0.25, 0.35, 0.45]:
        wt = 1.0 - ws
        rank_fused = ws * (ranks_s ** p_exp) + wt * (0.6 * (ranks_t0 ** p_exp) + 0.4 * (ranks_tema ** p_exp))
        scores_rank = snr_scores(rank_fused, labels, snrs, 36)
        print(f'Power p={p_exp:.1f}, Ws={ws:.2f} -> Top-1: {scores_rank[\"top1_accuracy\"]*100:6.2f}% | mAP: {scores_rank[\"mAP\"]*100:6.2f}% | mAP_macro: {scores_rank.get(\"mAP_macro\",0)*100:6.2f}% | AUC: {scores_rank[\"macro_auc\"]*100:6.2f}%')
"""

scratch_file = "eval_calibration_tta.py"
run_code = f"""
import subprocess
from pathlib import Path
Path('/marimo/imp/kd-v5/{scratch_file}').write_text({repr(code)}, encoding='utf-8')
res = subprocess.run(['python', '/marimo/imp/kd-v5/{scratch_file}'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')
print(res.stdout)
print(res.stderr)
"""

print("Running Calibration and Scaling Exploration on Molab...")
out = client.run(run_code)
sys.stdout.buffer.write(out.encode("utf-8"))
