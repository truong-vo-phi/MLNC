from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=180.0,
)

code = """
import sys
from pathlib import Path
import torch
from tqdm import tqdm

from src.model.beat.checkpoint import load_fold_teacher
from src.metrics import snr_scores
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t1, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_fold0_weights.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
t2, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_ema_standalone.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
t1.eval()
t2.eval()

base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

probs_list, labels_list, snrs_list = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc='Ensemble Teacher Eval'):
        wave = batch['waveform'].to(device, non_blocking=True)
        rms = wave.square().mean(dim=-1).sqrt().clamp_min(1e-8)
        norm_wave = wave * (0.07925 / rms)[:, None]
        l1 = t1(norm_wave).logits
        l2 = t2(norm_wave).logits
        logits = 0.5 * l1 + 0.5 * l2
        probs_list.append(logits.softmax(dim=-1).cpu())
        labels_list.append(batch['target'].argmax(dim=1))
        snrs_list.append(batch['snr_db'])

probs = torch.cat(probs_list).numpy()
labels = torch.cat(labels_list).numpy()
snrs = torch.cat(snrs_list).numpy()
scores = snr_scores(probs, labels, snrs, 36)
print(f"ENSEMBLE | Top-1: {scores['top1_accuracy']*100:.2f}% | Top-3: {scores['top3_accuracy']*100:.2f}% | Bal Acc: {scores['balanced_accuracy']*100:.2f}% | mAP: {scores['mAP']*100:.2f}% | AUC: {scores['macro_auc']*100:.2f}%")
"""

out = client.run(f"import subprocess\nPath('/marimo/imp/kd-v5/eval_ensemble.py').write_text({repr(code)})\nres = subprocess.run(['python', '/marimo/imp/kd-v5/eval_ensemble.py'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')\nprint(res.stdout)\nprint(res.stderr)")
sys.stdout.buffer.write(out.encode("utf-8"))
