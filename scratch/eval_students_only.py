"""
Evaluate purely student-only models on the full 3,064 validation clips.
BEATs is completely excluded at inference time (0.0 weight).
"""
import sys, json, torch, yaml
from pathlib import Path
import numpy as np

sys.path.insert(0, '/marimo/moe')
from scripts.load_dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.model.student import build_student
from src.metrics import snr_scores
from src.pipeline import run_tta_predict, softmax

def eval_student_pipeline(low_ckpt, high_ckpt, threshold_db, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_cfg = yaml.safe_load(open('/marimo/moe/config/kd_ced_tiny.yaml'))
    
    # Build models
    low_model = build_student(raw_cfg, root=Path('/marimo/moe/config')).to(device).eval()
    low_model.load_state_dict(torch.load(low_ckpt, map_location=device)['model'])
    
    high_model = build_student(raw_cfg, root=Path('/marimo/moe/config')).to(device).eval()
    high_model.load_state_dict(torch.load(high_ckpt, map_location=device)['model'])
    
    # Dataset
    base = Mixture36Dataset(root=Path('/marimo/dataset/mix-dataset-v2'), split='validation', duration=4.0)
    cache_dataset = NoiseCacheDataset(base, Path('/marimo/dataset/cache/validation.npy'), Path('/marimo/dataset/cache/validation.done'))
    loader = build_dataloader(cache_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_probs = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for batch in loader:
            wave = batch["waveform"].to(device)
            snrs = batch["snr_db"].numpy()
            targets = batch["target"].argmax(dim=-1).numpy()
            
            is_high = snrs >= threshold_db
            probs = np.zeros((len(wave), 36), dtype=np.float32)
            
            if not np.all(is_high):
                low_mask = ~is_high
                s_low_logits = run_tta_predict(low_model, wave[low_mask], device)
                probs[low_mask] = softmax(s_low_logits.cpu().numpy(), temp=1.40)
                
            if np.any(is_high):
                high_mask = is_high
                s_high_logits = run_tta_predict(high_model, wave[high_mask], device)
                probs[high_mask] = softmax(s_high_logits.cpu().numpy(), temp=1.40)
                
            all_probs.append(probs)
            all_labels.append(targets)
            all_snrs.append(snrs)
            
    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    snrs_arr = np.concatenate(all_snrs, axis=0)
    
    scores = snr_scores(probs_arr, labels_arr, snrs_arr, 36)
    return scores

# Run 1 Dual-Student (no teachers)
r1_low_ckpt = '/marimo/output/moe-from-scratch/20261105-081718-fp32-math/low/last_best_checkpoint.pt'
r1_high_ckpt = '/marimo/output/moe-from-scratch/20261105-081718-fp32-math/high/last_best_checkpoint.pt'

# Run 3 Dual-Student (no teachers)
r3_low_ckpt = '/marimo/output/moe-split-10-15/low/last_best_checkpoint.pt'
r3_high_ckpt = '/marimo/output/moe-split-10-15/high/last_best_checkpoint.pt'

res_r1_dual = eval_student_pipeline(r1_low_ckpt, r1_high_ckpt, 18.0, "Run 1 Dual-Student Only (tau=18.0)")
res_r3_dual = eval_student_pipeline(r3_low_ckpt, r3_high_ckpt, 12.5, "Run 3 Dual-Student Only (tau=12.5)")
res_r3_low_alone = eval_student_pipeline(r3_low_ckpt, r3_low_ckpt, 999.0, "Run 3 Low-Student Alone (all SNRs)")
res_r3_high_alone = eval_student_pipeline(r3_high_ckpt, r3_high_ckpt, -999.0, "Run 3 High-Student Alone (all SNRs)")

summary = {
    "run1_dual_student_only": res_r1_dual,
    "run3_dual_student_only": res_r3_dual,
    "run3_low_student_alone": res_r3_low_alone,
    "run3_high_student_alone": res_r3_high_alone,
}

Path('/marimo/output/student_only_benchmark.json').write_text(json.dumps(summary, indent=2))
print("EVAL_COMPLETE")
print("R1 Dual Student Top-1:", res_r1_dual['top1_accuracy'], "mAP:", res_r1_dual['mAP'], "F1:", res_r1_dual['macro_f1'])
print("R3 Dual Student Top-1:", res_r3_dual['top1_accuracy'], "mAP:", res_r3_dual['mAP'], "F1:", res_r3_dual['macro_f1'])
print("R3 Low Alone Top-1:", res_r3_low_alone['top1_accuracy'], "mAP:", res_r3_low_alone['mAP'])
print("R3 High Alone Top-1:", res_r3_high_alone['top1_accuracy'], "mAP:", res_r3_high_alone['mAP'])
