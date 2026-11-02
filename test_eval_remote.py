import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

code = """
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.split_moe import CEDSingleMoESplitFFN
from src.metrics import compute_snr_breakdown

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load model
model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=True).to(device).eval()

# Load validation data
val_data = np.load('/marimo/datasets/validation.npy', mmap_mode='r')
df_val = pd.read_csv('/marimo/datasets/mix-dataset-v2/eval/manifest.csv')
targets = np.array([json.loads(x) for x in df_val['multi_hot_36']])
target_labels = np.argmax(targets, axis=-1)
snrs = df_val['target_snr_db'].to_numpy()

batch_size = 128
num_samples = len(val_data)
all_preds = []
all_alphas = []

with torch.no_grad():
    for start_idx in tqdm(range(0, num_samples, batch_size), desc='Evaluating Validation Set', unit='batch'):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_wave = torch.from_numpy(np.array(val_data[start_idx:end_idx])).to(device)
        out = model(batch_wave, return_dict=True)
        probs = torch.softmax(out['logits'], dim=-1).cpu().numpy()
        alphas = out['alpha'].squeeze(-1).cpu().numpy()
        all_preds.append(probs)
        all_alphas.append(alphas)

preds = np.concatenate(all_preds, axis=0)
alphas = np.concatenate(all_alphas, axis=0)

results = compute_snr_breakdown(preds, target_labels, snrs, alphas)
print('\\n' + '='*80)
print('BẢNG KẾT QUẢ ĐÁNH GIÁ 9 CHỈ SỐ THEO TỪNG MỐC SNR VÀ TỔNG THỂ (ALL)')
print('='*80)
header = f\"{'SNR':<7} | {'Top-1 (%)':<9} | {'Top-3 (%)':<9} | {'Bal Acc':<9} | {'mAP (%)':<9} | {'AUC (%)':<9} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'F1 (%)':<9} | {'Alpha':<7}\"
print(header)
print('-'*len(header))

for snr_k in ['-5dB', '0dB', '5dB', '10dB', '15dB', '20dB', 'all']:
    res = results[snr_k]
    row = (
        f\"{snr_k:<7} | \"
        f\"{res['top1']*100:<9.2f} | \"
        f\"{res['top3']*100:<9.2f} | \"
        f\"{res['balanced_acc']*100:<9.2f} | \"
        f\"{res['map']*100:<9.2f} | \"
        f\"{res['auc']*100:<9.2f} | \"
        f\"{res['precision']*100:<9.2f} | \"
        f\"{res['recall']*100:<9.2f} | \"
        f\"{res['f1']*100:<9.2f} | \"
        f\"{res['mean_alpha']:<7.4f}\"
    )
    print(row)
print('='*80)
"""

rc = execute_remote(code)
print(f"Finished with code {rc}")
