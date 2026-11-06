import os, sys, json, yaml
from pathlib import Path
import numpy as np
import torch

MOE_PROJECT_ROOT = Path("/marimo/moe")
if str(MOE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOE_PROJECT_ROOT))

from src.pipeline import load_pipeline, evaluate_pipeline, ChampionPipeline
from src.metrics import snr_scores
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("/datasets/mix-dataset-v2")
cache_dir = Path("/datasets/cache-frcrn")

print("--- Evaluating Full MoE Pipeline: Run 3 (New Split: Low <=10 dB, High >=15 dB, tau=12.5 dB) ---")

# Load pipeline with threshold = 12.5 dB
pipe3 = load_pipeline(
    config_path=MOE_PROJECT_ROOT / "config/moe.yaml",
    low_ckpt="/marimo/output/moe-split-10-15/low/last_best_checkpoint.pt",
    high_ckpt="/marimo/output/moe-split-10-15/high/last_best_checkpoint.pt",
    t_fold0_ckpt="/marimo/pretrained/beats_teacher_fold0_weights.pt",
    t_ema_ckpt="/marimo/pretrained/beats_teacher_ema_standalone.pt",
    device=device,
)
pipe3.snr_threshold = 12.5
pipe3.router.threshold = 12.5

base = Mixture36Dataset(root=data_root, split="validation", duration=4.0)
cache_file = cache_dir / "validation.npy"
done_file = cache_dir / "validation.done"
cache_dataset = NoiseCacheDataset(base, cache_file, done_file)
loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)

all_probs = []
all_labels = []
all_snrs = []

for batch in tqdm(loader, desc="Evaluating Run 3"):
    wave = batch["waveform"]
    snrs = batch["snr_db"].numpy()
    targets = batch["target"].argmax(dim=-1).numpy()
    
    probs = pipe3.predict_batch(wave, snrs)
    all_probs.append(probs)
    all_labels.append(targets)
    all_snrs.append(snrs)

probs_arr = np.concatenate(all_probs, axis=0)
labels_arr = np.concatenate(all_labels, axis=0)
snrs_arr = np.concatenate(all_snrs, axis=0)

scores_run3 = snr_scores(probs_arr, labels_arr, snrs_arr, 36)

# Save result
Path("/marimo/output/run3_scores.json").write_text(json.dumps(scores_run3, indent=2), encoding="utf-8")
print("\nRUN 3 OVERALL TOP-1 ACCURACY:", scores_run3["top1_accuracy"] * 100)
print("RUN 3 BALANCED ACCURACY:", scores_run3["balanced_accuracy"] * 100)
print("RUN 3 MACRO F1:", scores_run3["macro_f1"] * 100)
print("RUN 3 mAP:", scores_run3["mAP"] * 100)
