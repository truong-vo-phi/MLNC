import sys
import time
sys.path.insert(0, "/marimo/moe-v2")
import torch
from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.models import CEDTinyBaseline
from src.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_base = Mixture36Dataset(root="/marimo/datasets/mix-dataset-v2", split="train")
train_cache = NoiseCacheDataset(train_base, "/marimo/datasets/train.npy")
train_loader = build_dataloader(train_cache, batch_size=128, shuffle=True, num_workers=0)

val_base = Mixture36Dataset(root="/marimo/datasets/mix-dataset-v2", split="eval")
val_cache = NoiseCacheDataset(val_base, "/marimo/datasets/validation.npy")
val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=0)

print("Instantiating CEDTinyBaseline...")
m = CEDTinyBaseline(num_classes=36).to(device)

trainer = Trainer(
    model=m,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=1,
    lr_backbone=5e-5,
    mixup_alpha=0.4,
    mixup_prob=0.8,
)

t0 = time.time()
res = trainer.fit()
t1 = time.time()
print(f"1 epoch of training on 30,240 samples took {t1 - t0:.2f} seconds!")
print("Epoch 1 Val Results:", res)
