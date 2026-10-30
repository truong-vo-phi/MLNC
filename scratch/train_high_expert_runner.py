from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=600.0,
)

code = """
import os, sys, math, time, yaml, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from main import build_student
from src.model.beat import load_fold_teacher
from src.metrics import snr_scores
from src.loss import LogitDistillationLoss, SoftTargetCrossEntropy
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.utils.files import rms_normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Config & Models
with open('/marimo/imp/kd-v5/config/kd_ced_tiny.yaml') as f:
    cfg = yaml.safe_load(f)

print('Initializing High-Expert for SNR 20 dB...')
high_student = build_student(cfg)
base_ckpt = '/marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt'
high_student.load_pretrained(base_ckpt)
high_student.to(device)

# Load Teacher EMA (best teacher at SNR 20 dB: 48.92%)
teacher, _ = load_fold_teacher('/marimo/pretrained/beats_teacher_ema_standalone.pt', '/marimo/imp/kd-v5/config/beat.yaml', device)
teacher.eval()

# 2. Datasets filtered for SNR 20
train_base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='train', duration=4.0)
train_snrs = [float(r['target_snr_db']) for r in train_base.rows]
train_idx_20 = [i for i, s in enumerate(train_snrs) if s == 20.0]

train_cache = NoiseCacheDataset(train_base, Path('/marimo/dataset/cache/train.npy'), Path('/marimo/dataset/cache/train.done'))
train_subset_20 = Subset(train_cache, train_idx_20)
train_loader = build_dataloader(train_subset_20, batch_size=32, shuffle=True, num_workers=0)

val_base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
val_snrs = [float(r['target_snr_db']) for r in val_base.rows]
val_idx_20 = [i for i, s in enumerate(val_snrs) if s == 20.0]

val_cache = NoiseCacheDataset(val_base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
val_subset_20 = Subset(val_cache, val_idx_20)
val_loader = build_dataloader(val_subset_20, batch_size=64, shuffle=False, num_workers=0)

print(f'Train SNR 20 samples: {len(train_idx_20)}, Val SNR 20 samples: {len(val_idx_20)}')

# 3. Training setup
epochs = 10
lr = 5e-5
weight_decay = 0.05
optimizer = AdamW(high_student.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
total_steps = epochs * len(train_loader)
warmup_steps = 100

def lr_schedule(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_schedule)
ce_loss = SoftTargetCrossEntropy(label_smoothing=0.1)
kd_loss = LogitDistillationLoss(temperature=4.0)

best_val_acc = 0.0
best_state = None
out_dir = Path('/marimo/output/high_expert')
out_dir.mkdir(parents=True, exist_ok=True)

# Initial eval before fine-tuning
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            wave = batch['waveform'].to(device, non_blocking=True)
            noise = rms_normalize(wave, 0.07925)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(noise).logits
            probs = logits.float().softmax(dim=-1).cpu()
            targets = batch['target'].argmax(dim=-1)
            correct += (probs.argmax(dim=-1) == targets).sum().item()
            total += len(targets)
            all_probs.append(probs)
            all_labels.append(targets)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = correct / total
    return acc, probs, labels

initial_acc, _, _ = evaluate(high_student, val_loader)
print(f'Initial Student Accuracy on SNR 20 dB: {initial_acc * 100:.2f}%')
best_val_acc = initial_acc

t0 = time.time()
for epoch in range(1, epochs + 1):
    high_student.train()
    total_loss, seen = 0.0, 0
    for batch in train_loader:
        wave = batch['waveform'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True).float()
        with torch.no_grad():
            noise = rms_normalize(wave, 0.07925)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                teacher_logits = teacher(noise).logits.float()
                
        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            s_logits = high_student(noise).logits.float()
            loss = ce_loss(s_logits, target) + 1.0 * kd_loss(s_logits, teacher_logits)
            
        loss.backward()
        nn.utils.clip_grad_norm_(high_student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * len(target)
        seen += len(target)
        
    val_acc, _, _ = evaluate(high_student, val_loader)
    lr_curr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:2d}/{epochs} | Train Loss: {total_loss/seen:.4f} | Val SNR 20 Acc: {val_acc*100:6.2f}% (Best: {best_val_acc*100:6.2f}%) | LR: {lr_curr:.2e}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model': high_student.state_dict(), 'best_acc': best_val_acc, 'epoch': epoch}, out_dir / 'last_best_checkpoint.pt')

elapsed = time.time() - t0
print(f'High-Expert training finished in {elapsed:.1f}s! Best Val SNR 20 Acc: {best_val_acc*100:.2f}%')
"""

scratch_file = "train_high_expert.py"
run_code = f"""
import subprocess
from pathlib import Path
Path('/marimo/imp/kd-v5/{scratch_file}').write_text({repr(code)}, encoding='utf-8')
res = subprocess.run(['python', '/marimo/imp/kd-v5/{scratch_file}'], capture_output=True, text=True, cwd='/marimo/imp/kd-v5')
print(res.stdout)
print(res.stderr)
"""

print("Sending High-Expert training job to Molab...")
out = client.run(run_code)
sys.stdout.buffer.write(out.encode("utf-8"))
