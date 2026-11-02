import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

code = """
import torch, numpy as np
from tqdm import tqdm
from src.dataset import Mixture36Dataset, KDNoiseCacheDataset, build_dataloader
from src.models.split_moe import CEDSingleMoESplitFFN
from src.metrics import compute_snr_breakdown

base_ds = Mixture36Dataset('datasets/mix-dataset-v2', split='eval')
val_ds = KDNoiseCacheDataset(base_ds, 'datasets/validation.npy')
val_loader = build_dataloader(val_ds, batch_size=128, shuffle=False, num_workers=2)

model = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=True).eval()

all_preds, all_alphas, all_targets, all_snrs = [], [], [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc='Inference Standalone MoE'):
        wave = batch['waveform']
        out = model(wave, return_dict=True)
        probs = torch.softmax(out['logits'], dim=-1).numpy()
        alpha = out['alpha'].numpy()
        all_preds.append(probs)
        all_alphas.append(alpha)
        all_targets.append(batch['target_idx'].numpy())
        all_snrs.append(batch['snr_db'].numpy())

preds = np.concatenate(all_preds, axis=0)
alphas = np.concatenate(all_alphas, axis=0)
targets = np.concatenate(all_targets, axis=0)
snrs = np.concatenate(all_snrs, axis=0)

results = compute_snr_breakdown(preds, targets, snrs, alphas=alphas)
print('Results computed successfully!')
for k, v in results.items():
    print(f"{k}: Top-1={v['top1']:.2%}, Top-3={v['top3']:.2%}, mAP={v['map']:.4f}, AUC={v['auc']:.4f}, Alpha={v['mean_alpha']:.4f}")
"""

execute_remote(code)
