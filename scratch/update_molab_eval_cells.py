import marimo._code_mode as cm

cell_vbla_code = '''import os
import json
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tqdm.auto import tqdm

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
METRICS_LIST = ["acc", "precision", "recall", "macro_f1", "micro_f1", "mAP"]

def compute_all_metrics(logits_or_probs: np.ndarray, targets: np.ndarray, num_classes: int = 36):
    targets_idx = np.argmax(targets, axis=-1) if targets.ndim == 2 else targets.astype(np.int64)
    if logits_or_probs.ndim == 2 and (np.any(logits_or_probs < 0) or np.any(logits_or_probs > 1.0)):
        exps = np.exp(logits_or_probs - np.max(logits_or_probs, axis=-1, keepdims=True))
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
    else:
        probs = logits_or_probs

    pred_idx = np.argmax(probs, axis=-1)
    acc = float(np.mean(pred_idx == targets_idx)) if len(targets_idx) > 0 else 0.0

    p, r, f, _ = precision_recall_fscore_support(targets_idx, pred_idx, labels=np.arange(num_classes), average="macro", zero_division=0)
    _, _, micro, _ = precision_recall_fscore_support(targets_idx, pred_idx, average="micro", zero_division=0)

    aps = [average_precision_score(targets_idx == c, probs[:, c]) for c in range(num_classes) if np.any(targets_idx == c)]
    mAP = float(np.mean(aps)) if aps else 0.0

    return {"acc": acc, "precision": float(p), "recall": float(r), "macro_f1": float(f), "micro_f1": float(micro), "mAP": mAP}

def compute_snr_breakdown(logits: np.ndarray, targets: np.ndarray, snrs: np.ndarray, num_classes: int = 36):
    breakdown = {}
    for snr_val in SNR_LEVELS:
        mask = np.isclose(snrs, float(snr_val), atol=0.5)
        if np.sum(mask) > 0:
            sub = compute_all_metrics(logits[mask], targets[mask], num_classes)
            sub["n"] = int(np.sum(mask))
            breakdown[f"{snr_val}dB"] = sub
    overall = compute_all_metrics(logits, targets, num_classes)
    overall["n"] = len(targets)
    breakdown["all"] = overall
    return breakdown
'''

async with cm.get_context() as ctx:
    ctx.edit_cell("vblA", code=cell_vbla_code, hide_code=False)
    ctx.run_cell("vblA")

print("Successfully updated cell vblA!")
