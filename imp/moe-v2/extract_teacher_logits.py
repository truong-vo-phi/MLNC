import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = Path("/marimo/moe-v2") if Path("/marimo/moe-v2").exists() else Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.metrics import compute_snr_breakdown
from src.models.beat import BEATClassifier, load_fold_teacher

def tta_forward(model, wave, device, shift_samples=0, rms=0.07925):
    if shift_samples != 0:
        if shift_samples > 0:
            w = F.pad(wave[:, shift_samples:], (0, shift_samples), mode="reflect")
        else:
            s = abs(shift_samples)
            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
    else:
        w = wave

    # RMS normalization
    eps = 1e-8
    rms_val = w.square().mean(dim=-1).sqrt()
    noise = w * (rms / (rms_val + eps))[:, None]

    with torch.autocast(device.type, dtype=torch.bfloat16):
        logits = model(noise).logits
    return logits.float()

def extract_logits_for_loader(model, loader, device, use_tta=True):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            wave = batch["waveform"].to(device, non_blocking=True)
            if use_tta:
                l0 = tta_forward(model, wave, device, shift_samples=0)
                lp = tta_forward(model, wave, device, shift_samples=800)   # +50ms
                ln = tta_forward(model, wave, device, shift_samples=-800)  # -50ms
                logits = (l0 + 0.5 * lp + 0.5 * ln) / 2.0
            else:
                logits = tta_forward(model, wave, device, shift_samples=0)
            all_logits.append(logits.cpu().numpy())
            if (batch_idx + 1) % 60 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1:3d}/{len(loader)} batches...", flush=True)
    return np.concatenate(all_logits, axis=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== EXTRACTING 3-WAY TTA TEACHER LOGITS ON {device} ===")

    data_root = Path("/marimo/datasets/mix-dataset-v2")
    cache_dir = Path("/marimo/datasets")
    out_dir = Path("/marimo/datasets/teacher_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_base = Mixture36Dataset(root=data_root, split="train")
    train_cache = NoiseCacheDataset(train_base, cache_dir / "train.npy")
    train_loader = build_dataloader(train_cache, batch_size=128, shuffle=False, num_workers=0)

    val_base = Mixture36Dataset(root=data_root, split="eval")
    val_cache = NoiseCacheDataset(val_base, cache_dir / "validation.npy")
    val_loader = build_dataloader(val_cache, batch_size=128, shuffle=False, num_workers=0)

    p_fold0 = "/marimo/pretrained/beats_teacher_fold0_weights.pt"
    p_ema = "/marimo/pretrained/beats_teacher_ema_standalone.pt"
    cfg_beat = "/marimo/moe-v2/config/beat.yaml"

    # 1. Fold0 Teacher
    f_val_t0 = out_dir / "val_t0_logits.npy"
    f_train_t0 = out_dir / "train_t0_logits.npy"
    if f_val_t0.is_file() and f_train_t0.is_file():
        print("\nLoading existing Fold0 logits from disk...")
        val_t0 = np.load(f_val_t0)
        train_t0 = np.load(f_train_t0)
        print(f"Loaded val_t0: {val_t0.shape}, train_t0: {train_t0.shape}")
    else:
        print(f"\nLoading Fold0 Teacher from {p_fold0}...")
        t_fold0, _ = load_fold_teacher(p_fold0, cfg_beat, device)

        if not f_val_t0.is_file():
            print("Extracting Fold0 Validation logits...")
            val_t0 = extract_logits_for_loader(t_fold0, val_loader, device, use_tta=True)
            np.save(f_val_t0, val_t0)
            print("Saved val_t0_logits.npy, shape:", val_t0.shape)
        else:
            val_t0 = np.load(f_val_t0)

        if not f_train_t0.is_file():
            print("Extracting Fold0 Train logits...")
            t0_start = time.time()
            train_t0 = extract_logits_for_loader(t_fold0, train_loader, device, use_tta=True)
            np.save(f_train_t0, train_t0)
            print(f"Saved train_t0_logits.npy, shape: {train_t0.shape} in {time.time() - t0_start:.1f}s")
        else:
            train_t0 = np.load(f_train_t0)
        del t_fold0
        torch.cuda.empty_cache()

    # 2. EMA Teacher
    f_val_tema = out_dir / "val_tema_logits.npy"
    f_train_tema = out_dir / "train_tema_logits.npy"
    if f_val_tema.is_file() and f_train_tema.is_file():
        print("\nLoading existing EMA logits from disk...")
        val_tema = np.load(f_val_tema)
        train_tema = np.load(f_train_tema)
        print(f"Loaded val_tema: {val_tema.shape}, train_tema: {train_tema.shape}")
    else:
        print(f"\nLoading EMA Teacher from {p_ema}...")
        t_ema, _ = load_fold_teacher(p_ema, cfg_beat, device)

        if not f_val_tema.is_file():
            print("Extracting EMA Validation logits...")
            val_tema = extract_logits_for_loader(t_ema, val_loader, device, use_tta=True)
            np.save(f_val_tema, val_tema)
            print("Saved val_tema_logits.npy, shape:", val_tema.shape)
        else:
            val_tema = np.load(f_val_tema)

        if not f_train_tema.is_file():
            print("Extracting EMA Train logits...")
            tema_start = time.time()
            train_tema = extract_logits_for_loader(t_ema, train_loader, device, use_tta=True)
            np.save(f_train_tema, train_tema)
            print(f"Saved train_tema_logits.npy, shape: {train_tema.shape} in {time.time() - tema_start:.1f}s")
        else:
            train_tema = np.load(f_train_tema)
        del t_ema
        torch.cuda.empty_cache()

    # Verify accuracy of extracted logits on validation set
    val_targets = np.array([val_cache[i]["target_idx"] for i in range(len(val_cache))])
    val_snrs = np.array([val_cache[i]["snr_db"] for i in range(len(val_cache))])

    p0 = F.softmax(torch.from_numpy(val_t0) / 1.20, dim=-1).numpy()
    p_ema_probs = F.softmax(torch.from_numpy(val_tema) / 1.20, dim=-1).numpy()

    m0 = compute_snr_breakdown(p0, val_targets, val_snrs, num_classes=36)
    m_ema = compute_snr_breakdown(p_ema_probs, val_targets, val_snrs, num_classes=36)
    print(f"\nFold0 TTA Val Acc: {m0['top1_accuracy']*100:.2f}% | mAP: {m0['mAP_macro']*100:.2f}%")
    print(f"EMA TTA Val Acc:   {m_ema['top1_accuracy']*100:.2f}% | mAP: {m_ema['mAP_macro']*100:.2f}%")

    # Dynamic SNR Blending
    is_high = val_snrs >= 18.0
    p_blend = np.zeros_like(p0)
    p_blend[~is_high] = 0.60 * p0[~is_high] + 0.40 * p_ema_probs[~is_high]
    p_blend[is_high] = 0.40 * p0[is_high] + 0.60 * p_ema_probs[is_high]
    m_blend = compute_snr_breakdown(p_blend, val_targets, val_snrs, num_classes=36)
    print(f"Dual-Teacher TTA Blended Val Acc: {m_blend['top1_accuracy']*100:.2f}% | mAP: {m_blend['mAP_macro']*100:.2f}%")
    print("ALL TEACHER LOGITS CACHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
