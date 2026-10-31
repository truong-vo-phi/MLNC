import sys
import torch
import numpy as np
from pathlib import Path

# Fix path
sys.path.insert(0, "/marimo/moe-v2")
from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.models.beat import BEATClassifier, load_fold_teacher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== VERIFY TEACHER MODELS ON SANDBOX ===")
print("Device:", device)

data_root = Path("/marimo/datasets/mix-dataset-v2")
cache_dir = Path("/marimo/datasets")

val_base = Mixture36Dataset(root=data_root, split="eval")
val_cache = NoiseCacheDataset(val_base, cache_dir / "validation.npy")
val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=0)
print(f"Validation dataset: {len(val_cache)} samples")

# Test Teacher Fold0
p_fold0 = "/marimo/pretrained/beats_teacher_fold0_weights.pt"
cfg_beat = "/marimo/moe-v2/config/beat.yaml"
print(f"\nLoading Teacher Fold0 from {p_fold0}...")
t_fold0, meta0 = load_fold_teacher(p_fold0, cfg_beat, device)
print("Fold0 loaded successfully! Meta:", meta0)

# Test Teacher EMA
p_ema = "/marimo/pretrained/beats_teacher_ema_standalone.pt"
print(f"\nLoading Teacher EMA from {p_ema}...")
t_ema, meta_ema = load_fold_teacher(p_ema, cfg_beat, device)
print("EMA loaded successfully! Meta:", meta_ema)

def eval_teacher(model, name):
    model.eval()
    all_probs, all_targets, all_snrs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wave = batch["waveform"].to(device, non_blocking=True)
            target = batch["target_idx"].numpy()
            snr_db = batch["snr_db"].numpy()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(wave).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(target)
            all_snrs.append(snr_db)
    probs_arr = np.concatenate(all_probs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)
    metrics = compute_snr_breakdown(probs_arr, targets_arr, snrs_arr, num_classes=36)
    print(f"[{name}] Top-1 Acc: {metrics['top1_accuracy']*100:.2f}% | mAP: {metrics['mAP_macro']*100:.2f}% | AUC: {metrics['macro_auc']*100:.2f}%")
    for snr in [-5, 0, 5, 10, 15, 20]:
        print(f"  SNR {snr:3d} dB: Acc={metrics.get(f'acc_snr_{snr}', 0)*100:.2f}% | mAP={metrics.get(f'mAP_snr_{snr}', 0)*100:.2f}%")
    return probs_arr, metrics

p0, m0 = eval_teacher(t_fold0, "BEATs Fold0")
p_ema_arr, m_ema = eval_teacher(t_ema, "BEATs EMA")

# Evaluate Simple Ensemble of the two teachers
p_ens = 0.5 * p0 + 0.5 * p_ema_arr
targets_arr = np.array([val_cache[i]["target_idx"] for i in range(len(val_cache))])
snrs_arr = np.array([val_cache[i]["snr_db"] for i in range(len(val_cache))])
m_ens = compute_snr_breakdown(p_ens, targets_arr, snrs_arr, num_classes=36)
print(f"\n[Dual-Teacher 50/50 Ensemble] Top-1 Acc: {m_ens['top1_accuracy']*100:.2f}% | mAP: {m_ens['mAP_macro']*100:.2f}%")
