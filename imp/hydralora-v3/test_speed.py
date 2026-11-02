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

data_root = Path("/marimo/datasets/mix-dataset-v2")
cache_dir = Path("/marimo/datasets")

train_base = Mixture36Dataset(root=data_root, split="train")
train_dataset = KDNoiseCacheDataset(
    train_base,
    cache_dir / "train.npy",
    t0_logits_path=cache_dir / "teacher_cache" / "train_t0_logits.npy",
    tema_logits_path=cache_dir / "teacher_cache" / "train_tema_logits.npy",
)

# Test num_workers=0, batch_size=128
loader = build_dataloader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CEDHydraLoRAV3().to(device)

model.train()
# Warmup 2 batches
it = iter(loader)
for _ in range(2):
    b = next(it)
    w = b["waveform"].to(device, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(w)

torch.cuda.synchronize()
t0 = time.time()
num_test = 10
for _ in range(num_test):
    b = next(it)
    w = b["waveform"].to(device, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(w)
torch.cuda.synchronize()
elapsed = time.time() - t0

per_batch = elapsed / num_test
total_batches = len(train_dataset) // 128
print(f"Batch size 128 with num_workers=0:")
print(f"{num_test} batches took {elapsed:.3f}s -> {per_batch:.4f}s per batch ({128/per_batch:.1f} samples/sec).")
print(f"Total batches per epoch: {total_batches}. Total epoch time: {total_batches * per_batch:.1f}s (~{total_batches * per_batch / 60:.2f} minutes)!")
