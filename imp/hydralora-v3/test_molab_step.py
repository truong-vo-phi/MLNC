import sys
for k in list(sys.modules.keys()):
    if k.startswith("src"):
        del sys.modules[k]
sys.path.insert(0, "/marimo/imp/hydralora-v3")

from pathlib import Path
import time
import torch

from src.dataset import KDNoiseCacheDataset, Mixture36Dataset, build_dataloader
from src.models.moe_hydra import CEDHydraLoRAV3
from src.kd_trainer import ReliabilityGatedKDTrainer

data_root = Path("/marimo/datasets/mix-dataset-v2")
cache_dir = Path("/marimo/datasets")

train_base = Mixture36Dataset(root=data_root, split="train")
val_base = Mixture36Dataset(root=data_root, split="eval")

train_dataset = KDNoiseCacheDataset(
    train_base,
    cache_dir / "train.npy",
    t0_logits_path=cache_dir / "teacher_cache" / "train_t0_logits.npy",
    tema_logits_path=cache_dir / "teacher_cache" / "train_tema_logits.npy",
)
val_dataset = KDNoiseCacheDataset(
    val_base,
    cache_dir / "validation.npy",
    t0_logits_path=cache_dir / "teacher_cache" / "val_t0_logits.npy",
    tema_logits_path=cache_dir / "teacher_cache" / "val_tema_logits.npy",
)

train_loader = build_dataloader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = build_dataloader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CEDHydraLoRAV3().to(device)

print(f"Model instantiated. Params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")
print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

trainer = ReliabilityGatedKDTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=1,
)

# Test 5 batches
model.train()
t0 = time.time()
for i, batch in enumerate(train_loader):
    if i >= 5:
        break
    wave = batch["waveform"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    target_idx = batch["target_idx"].to(device, non_blocking=True)
    snr_db = batch["snr_db"].to(device, non_blocking=True)
    t0_logits = batch["t0_logits"].to(device, non_blocking=True)
    tema_logits = batch["tema_logits"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(wave, return_dict=True)
        loss = trainer.bce_loss(out["logits"], target)
    trainer.scaler.scale(loss).backward()
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad()
    print(f"Batch {i+1} done. Loss: {loss.item():.4f}")

elapsed = time.time() - t0
print(f"5 batches executed in {elapsed:.2f}s ({elapsed/5:.4f}s per batch).")
print(f"Projected epoch time (472 batches): {472 * (elapsed/5):.1f}s (~{472 * (elapsed/5)/60:.1f} minutes).")
