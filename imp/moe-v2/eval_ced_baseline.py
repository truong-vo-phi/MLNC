import sys
import time
sys.path.insert(0, "/marimo/moe-v2")
import torch
from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.models import CEDTinyBaseline
from run_experiments import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_base = Mixture36Dataset(root="/marimo/datasets/mix-dataset-v2", split="eval")
val_cache = NoiseCacheDataset(val_base, "/marimo/datasets/validation.npy")
val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=4)

print("Instantiating CEDTinyBaseline...")
m = CEDTinyBaseline(num_classes=36).to(device)

t0 = time.time()
scores = evaluate_model(m, val_loader, device)
t1 = time.time()
print(f"Evaluation of 3,064 clips finished in {t1 - t0:.2f} seconds!")
print(f"Top-1 Acc: {scores['top1_accuracy'] * 100:.2f}% | Macro-Acc: {scores['macro_acc'] * 100:.2f}% | Macro mAP: {scores['mAP_macro'] * 100:.2f}%")
for snr in [-5, 0, 5, 10, 15, 20]:
    print(f"  SNR {snr:2d} dB: Acc = {scores.get(f'acc_snr_{snr}', 0.0) * 100:.2f}%, mAP = {scores.get(f'mAP_snr_{snr}', 0.0) * 100:.2f}%")
