import scipy.io.wavfile as wavfile
import numpy as np

val_arr = np.load('/marimo/datasets/mix-frcrn/validation.npy', mmap_mode='r')
sr, raw_audio = wavfile.read('/marimo/datasets/mix-dataset-v2/eval/mixture/validation_00030240.wav')

# normalize raw_audio to float [-1, 1] if int
if raw_audio.dtype == np.int16:
    raw_audio = raw_audio.astype(np.float32) / 32768.0
elif raw_audio.dtype == np.int32:
    raw_audio = raw_audio.astype(np.float32) / 2147483648.0
elif raw_audio.dtype != np.float32:
    raw_audio = raw_audio.astype(np.float32)

frcrn_audio = np.array(val_arr[0], dtype=np.float32)

raw_rms = float(np.sqrt(np.mean(raw_audio**2)))
frcrn_rms = float(np.sqrt(np.mean(frcrn_audio**2)))
diff = raw_audio - frcrn_audio
diff_rms = float(np.sqrt(np.mean(diff**2)))

corr = float(np.corrcoef(raw_audio, frcrn_audio)[0, 1])

print("=== RAW vs FRCRN COMPARISON FOR FIRST CLIP (SNR = -5dB) ===")
print(f"Raw mixture: RMS = {raw_rms:.5f}, Peak = {np.abs(raw_audio).max():.4f}")
print(f"FRCRN audio: RMS = {frcrn_rms:.5f}, Peak = {np.abs(frcrn_audio).max():.4f}")
print(f"Difference (Raw - FRCRN, estimated speech component): RMS = {diff_rms:.5f}")
print(f"Correlation between Raw and FRCRN: {corr:.4f}")
print(f"Ratio FRCRN_RMS / Raw_RMS: {frcrn_rms / (raw_rms + 1e-8):.4f}")

# Also check a clip at SNR = +20dB
import pandas as pd
df_val = pd.read_csv('/marimo/datasets/mix-dataset-v2/eval/manifest.csv')
idx_20 = df_val.index[df_val['target_snr_db'] == 20.0][0]
row_20 = df_val.iloc[idx_20]
raw_path_20 = f"/marimo/datasets/mix-dataset-v2/{row_20['mixture_path']}"
sr2, raw_audio_20 = wavfile.read(raw_path_20)
if raw_audio_20.dtype == np.int16:
    raw_audio_20 = raw_audio_20.astype(np.float32) / 32768.0
frcrn_20 = np.array(val_arr[idx_20], dtype=np.float32)

raw_rms_20 = float(np.sqrt(np.mean(raw_audio_20**2)))
frcrn_rms_20 = float(np.sqrt(np.mean(frcrn_20**2)))
diff_20 = raw_audio_20 - frcrn_20
diff_rms_20 = float(np.sqrt(np.mean(diff_20**2)))
corr_20 = float(np.corrcoef(raw_audio_20, frcrn_20)[0, 1])

print(f"\n=== RAW vs FRCRN COMPARISON FOR CLIP AT SNR = +20dB (Sample: {row_20['sample_id']}) ===")
print(f"Raw mixture: RMS = {raw_rms_20:.5f}, Peak = {np.abs(raw_audio_20).max():.4f}")
print(f"FRCRN audio: RMS = {frcrn_rms_20:.5f}, Peak = {np.abs(frcrn_20).max():.4f}")
print(f"Difference (Raw - FRCRN, estimated speech component): RMS = {diff_rms_20:.5f}")
print(f"Correlation between Raw and FRCRN: {corr_20:.4f}")
print(f"Ratio FRCRN_RMS / Raw_RMS: {frcrn_rms_20 / (raw_rms_20 + 1e-8):.4f}")
