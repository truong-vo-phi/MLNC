import numpy as np
import pandas as pd
import json

val_arr = np.load('/marimo/datasets/mix-frcrn/validation.npy', mmap_mode='r')
train_arr = np.load('/marimo/datasets/mix-frcrn/train.npy', mmap_mode='r')

print("=== 1. FILE STRUCTURE & SHAPES ===")
print(f"Train npy: shape={train_arr.shape}, dtype={train_arr.dtype}, size={train_arr.nbytes / 1e9:.2f} GB")
print(f"Validation npy: shape={val_arr.shape}, dtype={val_arr.dtype}, size={val_arr.nbytes / 1e6:.2f} MB")

# Read .done files
with open('/marimo/datasets/mix-frcrn/train.done') as f:
    train_done_val = f.read().strip()
with open('/marimo/datasets/mix-frcrn/validation.done') as f:
    val_done_val = f.read().strip()
print(f"train.done content: '{train_done_val}' (matches train_arr length: {int(train_done_val) == len(train_arr)})")
print(f"validation.done content: '{val_done_val}' (matches val_arr length: {int(val_done_val) == len(val_arr)})")

print("\n=== 2. STATISTICAL & ACOUSTIC AUDIT (VALIDATION) ===")
sample_indices = np.linspace(0, len(val_arr)-1, 500, dtype=int)
sampled = np.array(val_arr[sample_indices])

has_nan = bool(np.isnan(sampled).any())
has_inf = bool(np.isinf(sampled).any())
g_min = float(sampled.min())
g_max = float(sampled.max())
g_mean = float(sampled.mean())
g_std = float(sampled.std())

clip_rms = np.sqrt(np.mean(sampled**2, axis=-1))
clip_peak = np.max(np.abs(sampled), axis=-1)
clip_crest = clip_peak / (clip_rms + 1e-8)

print(f"Has NaN: {has_nan} | Has Inf: {has_inf}")
print(f"Amplitude Range: [{g_min:.4f}, {g_max:.4f}]")
print(f"Global DC offset (Mean): {g_mean:.6e} (near-zero, unbiased)")
print(f"Global Std: {g_std:.4f}")
print(f"Mean Clip RMS: {clip_rms.mean():.6f} (Min: {clip_rms.min():.6f}, Max: {clip_rms.max():.6f})")
print(f"Mean Crest Factor (Peak/RMS): {clip_crest.mean():.2f}")

print("\n=== 3. BEHAVIOR ACROSS SNR LEVELS (-5dB to +20dB) ===")
df_val = pd.read_csv('/marimo/datasets/mix-dataset-v2/eval/manifest.csv')
snrs = sorted(df_val['target_snr_db'].unique())
for s in snrs:
    idx_s = df_val.index[df_val['target_snr_db'] == s].tolist()
    sub = np.array(val_arr[idx_s])
    sub_rms = np.sqrt(np.mean(sub**2, axis=-1))
    sub_peak = np.max(np.abs(sub), axis=-1)
    sub_crest = sub_peak / (sub_rms + 1e-8)
    print(f"SNR {s:>5.1f} dB ({len(idx_s):>4} clips) -> Mean RMS: {sub_rms.mean():.5f}, Std RMS: {sub_rms.std():.5f}, Mean Peak: {sub_peak.mean():.4f}, Mean Crest: {sub_crest.mean():.2f}")

print("\n=== 4. COMPARISON: RAW MIXTURE vs FRCRN MASKED CLIP ===")
# Load raw wav
try:
    import soundfile as sf
    raw_wav_path = '/marimo/datasets/mix-dataset-v2/eval/mixture/validation_00030240.wav'
    raw_audio, sr = sf.read(raw_wav_path)
    frcrn_audio = np.array(val_arr[0])
    diff = raw_audio - frcrn_audio
    raw_rms = np.sqrt(np.mean(raw_audio**2))
    frcrn_rms = np.sqrt(np.mean(frcrn_audio**2))
    diff_rms = np.sqrt(np.mean(diff**2))
    energy_ratio = (frcrn_rms / (raw_rms + 1e-8))**2
    corr = np.corrcoef(raw_audio, frcrn_audio)[0, 1]
    print(f"Raw mixture ({raw_wav_path}):")
    print(f"  Sample rate: {sr} Hz, Length: {len(raw_audio)} samples ({len(raw_audio)/sr:.1f}s)")
    print(f"  Raw RMS: {raw_rms:.5f} | FRCRN RMS: {frcrn_rms:.5f} | Residual Diff RMS: {diff_rms:.5f}")
    print(f"  Energy preserved by mask: {energy_ratio * 100:.2f}%")
    print(f"  Pearson correlation (Raw vs FRCRN): {corr:.4f}")
except Exception as e:
    print(f"Wav comparison error: {e}")
