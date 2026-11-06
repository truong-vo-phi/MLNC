import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import MixtureDataset
from src.models.ced_moe_split import CEDSingleMoESplitFFN
from src.metrics import compute_9_metrics_dict, compute_snr_breakdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load config
with open("/marimo/config/config.yaml") as f:
    cfg = yaml.safe_load(f)

data_root = "/datasets/mix-dataset-v2"
batch_size = 8

# 2. Build datasets
ds_train = MixtureDataset(data_root, split="train")
ds_eval = MixtureDataset(data_root, split="eval")

# Smoke test subset: 16 samples train, 16 samples eval
smoke_train_loader = DataLoader(Subset(ds_train, range(16)), batch_size=batch_size, shuffle=True)
smoke_val_loader = DataLoader(Subset(ds_eval, range(16)), batch_size=batch_size, shuffle=False)

# 3. Model & Optimizer
model = CEDSingleMoESplitFFN(
    num_classes=36,
    load_pretrained=True,
    pretrained_path="/marimo/pretrained/model.safetensors"
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

# 4. Smoke-test Training Loop
model.train()
print("\n--- RUNNING SMOKE-TEST: TRAIN STEP ---")
train_pbar = tqdm(smoke_train_loader, desc="Smoke Train")
for batch in train_pbar:
    wav = batch["waveform"].to(device)
    target = batch["target"].to(device)
    snr_db = batch["snr_db"].to(device)
    target_idx = target.argmax(dim=-1)

    optimizer.zero_grad()
    out = model(wav, return_dict=True)
    loss_ce = criterion_ce(out["logits"], target_idx)
    loss_router = model.router.compute_router_loss(out["alpha"], snr_db)
    loss = 0.5 * loss_ce + 0.2 * loss_router

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "ce": f"{loss_ce.item():.4f}", "alpha": f"{out['alpha'].mean().item():.3f}"})

# 5. Smoke-test Eval Loop
model.eval()
print("\n--- RUNNING SMOKE-TEST: EVAL STEP ---")
all_logits, all_targets, all_snrs, all_alphas = [], [], [], []
val_pbar = tqdm(smoke_val_loader, desc="Smoke Eval")
with torch.no_grad():
    for batch in val_pbar:
        wav = batch["waveform"].to(device)
        target = batch["target"].to(device)
        snr_db = batch["snr_db"].to(device)
        out = model(wav, return_dict=True)
        all_logits.append(out["logits"].cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_snrs.append(snr_db.cpu().numpy())
        all_alphas.append(out["alpha"].cpu().numpy())

import numpy as np
logits_np = np.concatenate(all_logits, axis=0)
targets_np = np.concatenate(all_targets, axis=0)
snrs_np = np.concatenate(all_snrs, axis=0)
alphas_np = np.concatenate(all_alphas, axis=0)

metrics_9 = compute_9_metrics_dict(logits_np, targets_np, alphas=alphas_np, num_classes=36)
snr_res = compute_snr_breakdown(logits_np, targets_np, snrs_np, alphas=alphas_np, num_classes=36, use_9_metrics=True)

print("\n--- SMOKE-TEST METRICS ---")
for k, v in metrics_9.items():
    print(f"  {k}: {v:.4f}")

print("\n--- SNR BREAKDOWN ---")
for snr_tag, stats in snr_res.items():
    if stats.get("count", 0) > 0:
        print(f"  SNR {snr_tag}: count={stats['count']}, top1={stats['top1']:.2f}, mean_alpha={stats['mean_alpha']:.3f}")

print("\n>>> ALL SMOKE-TEST CHECKS PASSED WITH ZERO ERRORS! <<<")
