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

# 1. Load Models:
# - Low Expert: General CED-Tiny (15ep)
low_student = build_student(cfg)
low_ckpt = '/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt'
low_student.load_pretrained(low_ckpt)
low_student.to(device).eval()

# - High Expert: SNR 20 Fine-tuned CED-Tiny
high_student = build_student(cfg)
high_ckpt = '/marimo/output/high_expert/last_best_checkpoint.pt'
high_ckpt_data = torch.load(high_ckpt, map_location='cpu')
high_student.load_state_dict(high_ckpt_data['model'])
high_student.to(device).eval()

# - Teachers: Fold0 and EMA
t_fold0, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_fold0_weights.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
t_fold0.eval()

t_ema, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_ema_standalone.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
t_ema.eval()

# 2. Run Inference on all 3,064 validation clips
base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

low_s_probs, high_s_probs = [], []
t0_probs, tema_probs = [], []
labels, snrs = [], []

with torch.no_grad():
    for batch in tqdm(loader, desc='Inference all 4 models'):
        wave = batch['waveform'].to(device, non_blocking=True)
        noise = rms_normalize(wave, 0.07925)
        
        with torch.autocast(device.type, dtype=torch.bfloat16):
            l_s = low_student(noise).logits
            h_s = high_student(noise).logits
            l_t0 = t_fold0(noise).logits
            l_tema = t_ema(noise).logits
            
        low_s_probs.append(l_s.float().softmax(dim=-1).cpu().numpy())
        high_s_probs.append(h_s.float().softmax(dim=-1).cpu().numpy())
        t0_probs.append(l_t0.float().softmax(dim=-1).cpu().numpy())
        tema_probs.append(l_tema.float().softmax(dim=-1).cpu().numpy())
        labels.append(batch['target'].argmax(dim=-1).numpy())
        snrs.append(batch['snr_db'].numpy())

low_s_probs = np.concatenate(low_s_probs, axis=0)
high_s_probs = np.concatenate(high_s_probs, axis=0)
t0_probs = np.concatenate(t0_probs, axis=0)
tema_probs = np.concatenate(tema_probs, axis=0)
labels = np.concatenate(labels, axis=0)
snrs = np.concatenate(snrs, axis=0)

mask_20 = (snrs == 20.0)
mask_rest = (snrs != 20.0)

# Build configurations:
# Config 1: Baseline Student Alone (no expert routing)
p_student_alone = low_s_probs

# Config 2: Dual-Student Routing (Low-Student for rest, High-Student for SNR 20)
p_dual_student = np.zeros_like(low_s_probs)
p_dual_student[mask_rest] = low_s_probs[mask_rest]
p_dual_student[mask_20] = high_s_probs[mask_20]

# Config 3: Baseline General Ensemble (0.4 Low-Student + 0.6 Teacher Fold0) everywhere
p_gen_ens = 0.4 * low_s_probs + 0.6 * t0_probs

# Config 4: Dual-Expert Ensemble:
# - For rest (-5 to 15 dB): 0.4 Low-Student + 0.6 Teacher Fold0 (Best for low/med SNR: 65.99%)
# - For SNR 20 dB: 0.35 High-Student + 0.35 Teacher EMA + 0.30 Teacher Fold0 (Best for faint noise!)
p_dual_ens = np.zeros_like(low_s_probs)
p_dual_ens[mask_rest] = 0.40 * low_s_probs[mask_rest] + 0.60 * t0_probs[mask_rest]
p_dual_ens[mask_20] = 0.35 * high_s_probs[mask_20] + 0.35 * tema_probs[mask_20] + 0.30 * t0_probs[mask_20]

configs = [
    ('1. Student CED-Tiny Đơn lẻ (Không định tuyến)', p_student_alone),
    ('2. Dual-Student (Low-Student + High-Student SNR 20)', p_dual_student),
    ('3. General Ensemble (0.4 S + 0.6 T_fold0 - Đồng nhất)', p_gen_ens),
    ('4. Dual-Expert Ensemble (Low Expert + High Expert SNR 20)', p_dual_ens),
]

print('========================================================================================================')
print('KẾT QUẢ THỰC NGHIỆM: PHƯƠNG PHÁP HIGH-EXPERT (SNR 20 dB) + LOW-EXPERT (SNR CÒN LẠI)')
print('========================================================================================================')
for name, probs in configs:
    scores = snr_scores(probs, labels, snrs, 36)
    acc_all = scores['top1_accuracy'] * 100
    acc_rest = (probs[mask_rest].argmax(axis=-1) == labels[mask_rest]).mean() * 100
    acc_20 = (probs[mask_20].argmax(axis=-1) == labels[mask_20]).mean() * 100
    map_all = scores['mAP'] * 100
    auc_all = scores['macro_auc'] * 100
    top3_all = scores['top3_accuracy'] * 100
    print(f'{name:<56}')
    print(f'   -> OVERALL: Top-1: {acc_all:5.2f}% | Top-3: {top3_all:5.2f}% | mAP: {map_all:5.2f}% | AUC: {auc_all:5.2f}%')
    print(f'   -> Phân rã: Rest (-5..15 dB / 2555 clips): {acc_rest:5.2f}% | SNR 20 (509 clips): {acc_20:5.2f}%')
    print('--------------------------------------------------------------------------------------------------------')

# SNR Breakdown for Config 4
scores_de = snr_scores(p_dual_ens, labels, snrs, 36)
print('\\nCHI TIẾT TỪNG MỨC SNR CỦA DUAL-EXPERT ENSEMBLE (CONFIG 4):')
print('--------------------------------------------------------------------------------------------------------')
for snr in sorted([-5, 0, 5, 10, 15, 20]):
    t1 = scores_de.get(f'top1_accuracy_snr_{snr}', scores_de.get(f'acc_snr_{snr}', 0.0)) * 100
    t3 = scores_de.get(f'top3_accuracy_snr_{snr}', 0.0) * 100
    map_v = scores_de.get(f'mAP_snr_{snr}', 0.0) * 100
    auc = scores_de.get(f'macro_auc_snr_{snr}', 0.0) * 100
    print(f'SNR: {snr:<4} dB | Top-1: {t1:5.2f}% | Top-3: {t3:5.2f}% | mAP: {map_v:5.2f}% | AUC: {auc:5.2f}%')
print('========================================================================================================')
"""

scratch_file = "eval_dual_expert.py"
run_code = f"""
import subprocess
from pathlib import Path
Path('/marimo/imp/kd-v5/{scratch_file}').write_text({repr(code)}, encoding='utf-8')
res = subprocess.run(['python', '/marimo/imp/kd-v5/{scratch_file}'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')
print(res.stdout)
print(res.stderr)
"""

print("Executing Dual-Expert Evaluation Pipeline on Molab...")
out = client.run(run_code)
sys.stdout.buffer.write(out.encode("utf-8"))
