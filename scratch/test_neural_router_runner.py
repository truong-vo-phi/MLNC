from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=120.0,
)

code = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.model.student.frontend import LogMel, LogMelConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Feature Extractor: LogMel mean and std across time
frontend = LogMel()
frontend.to(device).eval()

# 2. Extract features for a subset of train (6000 clips) and all val (3064 clips)
train_base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='train', duration=4.0)
train_cache = NoiseCacheDataset(train_base, Path('/marimo/dataset/cache/train.npy'), Path('/marimo/dataset/cache/train.done'))
# sample balanced 6000 clips (1000 per SNR)
train_loader = build_dataloader(train_cache, batch_size=128, shuffle=True, num_workers=0)

val_base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
val_cache = NoiseCacheDataset(val_base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=0)

def extract_features(loader, max_batches=50):
    feats, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches: break
            wave = batch['waveform'].to(device)
            # LogMel: [B, 1, 64, 398]
            mel = frontend(wave).squeeze(1) # [B, 64, 398]
            # Summary statistics: mean [B, 64], std [B, 64], energy [B, 1]
            m = mel.mean(dim=-1)
            s = mel.std(dim=-1)
            rms = wave.pow(2).mean(dim=-1, keepdim=True).sqrt().log()
            f = torch.cat([m, s, rms], dim=-1) # [B, 129]
            
            # Target: 1 if SNR == 20 else 0
            y = (batch['snr_db'] == 20.0).long().to(device)
            feats.append(f)
            targets.append(y)
    return torch.cat(feats), torch.cat(targets)

print('Extracting features...')
X_train, y_train = extract_features(train_loader, max_batches=60) # ~7680 clips
X_val, y_val = extract_features(val_loader, max_batches=None)     # all 3064 clips

# Train a 2-layer Neural Router: Linear(129, 64) -> GELU -> Linear(64, 2)
router = nn.Sequential(
    nn.Linear(129, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 2)
).to(device)

optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device)) # class weight for balanced 1:5 ratio

for epoch in range(15):
    router.train()
    idx = torch.randperm(len(X_train))
    for b in range(0, len(X_train), 128):
        bx = X_train[idx[b:b+128]]
        by = y_train[idx[b:b+128]]
        out = router(bx)
        loss = criterion(out, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

router.eval()
with torch.no_grad():
    val_out = router(X_val)
    val_probs = val_out.softmax(dim=-1)[:, 1].cpu().numpy()
    val_preds = val_out.argmax(dim=-1).cpu().numpy()
    y_true = y_val.cpu().numpy()

acc = (val_preds == y_true).mean() * 100
pos_mask = (y_true == 1) # true SNR 20
recall_20 = (val_preds[pos_mask] == 1).mean() * 100
precision_20 = ((val_preds == 1) & (y_true == 1)).sum() / max(1, (val_preds == 1).sum()) * 100

print(f'Neural Router Parameters: {sum(p.numel() for p in router.parameters()):,}')
print(f'Neural Router Accuracy: {acc:.2f}% | Recall SNR 20: {recall_20:.2f}% | Precision SNR 20: {precision_20:.2f}%')
"""

out = client.run(f"import subprocess\nPath('/marimo/imp/kd-v5/test_neural_router.py').write_text({repr(code)})\nres = subprocess.run(['python', '/marimo/imp/kd-v5/test_neural_router.py'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')\nprint(res.stdout)\nprint(res.stderr)")
import sys
sys.stdout.buffer.write(out.encode("utf-8"))
