from __future__ import annotations

from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=180.0,
)

code = """
import subprocess
from pathlib import Path

eval_script = '''
import sys
from pathlib import Path
import torch
from tqdm import tqdm

from src.model.beat.checkpoint import load_fold_teacher
from src.metrics import snr_scores
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_ckpt = '/marimo/pretrained/beats_teacher_ema_standalone.pt'
teacher_cfg = '/marimo/imp/kd-v5/config/beat.yaml'

teacher, meta = load_fold_teacher(teacher_ckpt, teacher_cfg, device)
teacher.eval()

base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

probs_list, labels_list, snrs_list = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc='Eval Teacher EMA on Clean Val'):
        wave = batch['waveform'].to(device, non_blocking=True)
        rms = wave.square().mean(dim=-1).sqrt().clamp_min(1e-8)
        norm_wave = wave * (0.07925 / rms)[:, None]
        logits = teacher(norm_wave).logits
        probs_list.append(logits.softmax(dim=-1).cpu())
        labels_list.append(batch['target'].argmax(dim=1))
        snrs_list.append(batch['snr_db'])

probs = torch.cat(probs_list).numpy()
labels = torch.cat(labels_list).numpy()
snrs = torch.cat(snrs_list).numpy()

scores = snr_scores(probs, labels, snrs, 36)

print('=============================================================================')
print('KẾT QUẢ THỰC TẾ: TEACHER BEATS EMA STANDALONE TRÊN TẬP VAL CLEAN (3064 clips)')
print('=============================================================================')
print(f"OVERALL | Top-1: {scores['top1_accuracy']*100:6.2f}% | Top-3: {scores['top3_accuracy']*100:6.2f}% | Bal Acc: {scores['balanced_accuracy']*100:6.2f}% | mAP: {scores['mAP']*100:6.2f}% | AUC: {scores['macro_auc']*100:6.2f}%")
for snr in sorted([-5, 0, 5, 10, 15, 20]):
    t1 = scores.get(f'top1_accuracy_snr_{snr}', scores.get(f'acc_snr_{snr}', 0.0)) * 100
    t3 = scores.get(f'top3_accuracy_snr_{snr}', 0.0) * 100
    bacc = scores.get(f'balanced_accuracy_snr_{snr}', scores.get(f'balanced_snr_{snr}', 0.0)) * 100
    map_v = scores.get(f'mAP_snr_{snr}', 0.0) * 100
    auc = scores.get(f'macro_auc_snr_{snr}', 0.0) * 100
    print(f"SNR: {snr:<8} | Top-1: {t1:6.2f}% | Top-3: {t3:6.2f}% | Bal Acc: {bacc:6.2f}% | mAP: {map_v:6.2f}% | AUC: {auc:6.2f}%")
print('=============================================================================')
'''

Path('/marimo/imp/kd-v5/eval_teacher_ema_clean.py').write_text(eval_script, encoding='utf-8')
res = subprocess.run(['python', '/marimo/imp/kd-v5/eval_teacher_ema_clean.py'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')
print('STDOUT:', res.stdout)
print('STDERR:', res.stderr)
"""

print(client.run(code))
