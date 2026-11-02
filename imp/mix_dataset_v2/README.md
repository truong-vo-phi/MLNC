# Clean mix-dataset v2 experiments

Code used for the clean mixture-v2 experiments: nine CE-only baselines and the
team's dual-Teacher CED-MoE distillation pipeline. This directory contains source,
configuration, tests and required upstream licenses. Audio, pretrained/trained
weights, logits caches, logs, predictions and benchmark results are excluded.

## Environment

Run commands **from this directory**, using Python 3.12. The recorded environment
used PyTorch/TorchAudio 2.6.0 + CUDA 12.6, TorchVision 0.21.0 and timm 1.0.30.

```powershell
cd imp/mix_dataset_v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

## Dataset and machine-local paths

The dataset is pinned to Hugging Face revision
`ea3ce2572be476fa90b5f6424ab461330bd23b39` of
[phitruong0606/mix-dataset-v2](https://huggingface.co/datasets/phitruong0606/mix-dataset-v2).
There are 30,240 train and 3,064 validation (`eval`) samples, 36 classes and six
SNR values: −5, 0, 5, 10, 15 and 20 dB. No old validation allowlist is reapplied.

```powershell
python download_team_mix_v2.py
python configure_mix_v2.py --beats-checkpoint "D:/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" --check
```

The downloader verifies file size, LFS SHA-256, archive paths and ZIP CRC. It needs
space for both downloaded ZIPs and extracted mixture/FRCRN arrays; it does not
download the separate clean AudioSet archive. `cache-nhat.zip` contains FRCRN
residual-noise waveforms (`train.npy`, `validation.npy`), **not** raw mixtures.

BEATs CE fine-tuning requires the separate official **iter3+ AS2M cpt2** checkpoint
from [Microsoft BEATs](https://github.com/microsoft/unilm/tree/master/beats).
Point `--beats-checkpoint` at that existing file. The HF noise-trained Teachers
are different checkpoints and must not replace this baseline initialization.

To reuse downloaded data without copying it:

```powershell
python configure_mix_v2.py --data-root "D:/data/mix-dataset-v2" --cache-dir "D:/data/frcrn-cache" --weights-dir "D:/pretrained/team" --beats-checkpoint "D:/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" --check
```

This generates an ignored `TEAM_DATA_PATHS.json` under `benchmark_results/`.
The configuration checker reads train/eval only; test is not used for selection.

## Nine CE-only baselines

```powershell
python prepare_ce9.py
python train_ce9_clean.py
```

`prepare_ce9.py` downloads pinned CED weights, PANNs-CNN10 and EfficientAT-MN10
weights, initializes ImageNet models through timm, and checks forward/backward at
the actual training micro-batch plus batch-one evaluation for all nine models.
No model is silently replaced with a smaller or randomly initialized backbone.

The queue runs these models sequentially:

| Model | Pretrained domain | Micro-batch × accumulation | Backbone / head LR |
|---|---|---:|---:|
| PANNs-CNN10 | AudioSet | 32 × 2 | 3e−5 / 3e−4 |
| CED-Tiny | AudioSet | 32 × 2 | 3e−5 / 3e−4 |
| CED-Mini | AudioSet | 32 × 2 | 3e−5 / 3e−4 |
| CED-Small | AudioSet | 16 × 4 | 3e−5 / 3e−4 |
| EfficientAT MN10 | AudioSet | 32 × 2 | 3e−5 / 3e−4 |
| ViT-S/16 | ImageNet | 16 × 4 | 3e−5 / 3e−4 |
| MobileViT-S | ImageNet | 16 × 4 | 3e−5 / 3e−4 |
| MobileNetV3-Large | ImageNet | 32 × 2 | 3e−5 / 3e−4 |
| BEATs 12 blocks | AudioSet | 8 × 8 | 1e−5 / 1e−4 |

All use raw mixture WAVs, a new 36-class classifier, full-backbone fine-tuning and
unweighted CE on **raw logits**. No KD, FRCRN, Mixup, label smoothing, SpecAugment,
TTA or ensemble is added. Fixed STFT/Mel preprocessing stays fixed. CED uses the
original mean-pooling architecture, without the MoE experiment's CLS/GREP/router.

Shared settings: 15 epochs, seed 2026, effective batch 64, AdamW, weight decay
0.01, BF16, gradient clipping 1.0, two-epoch warmup then cosine decay. Select
`best.pt` by validation accuracy, retaining the earlier epoch on ties; no early
stopping. AudioSet/ImageNet initialization and frontend differences are explicit:
this compares transfer-learning configurations, not architecture alone.

Each epoch logs `train_loss`, `train_acc`, `train_mAP`, `val_loss`, `val_acc`,
`val_mAP`, and validation metrics for all six SNR values plus `all`: `acc`, macro
`precision`, macro `recall`, `macro_f1`, `micro_f1`, `mAP`. mAP uses softmax
probabilities. Train metrics aggregate online train-mode predictions while
weights change; validation uses fixed eval-mode weights at the epoch end.

Generated files are under `benchmark_results/ce9_clean_mixture_20261102/`:
per-model logs/history/checkpoints/predictions, parameter counts, per-SNR/per-class
CSVs, and the aggregate `BAO_CAO_CE9_VI.md`, `metrics_all_models.csv` and config
exports. Exact original training settings are in `config/ce9_original_run.json`.

Run only one queue on a GPU. Re-running the same command skips completed models
and resumes unfinished models from the last completed epoch. Errors are recorded
per model and the queue continues; inspect its final status for failures.

## Full Teacher logits and team CED-MoE KD

```powershell
python -m pytest runtime/kd_moe_ced/tests -q
python verify_team_frcrn_cache.py
python train_team_kd_moe_full.py
```

The verified team runtime is bundled from commit
`697064dfd638f0b934f1a9ee84cac2cf0f32c063`, with the previously tested TorchAudio,
Teacher-loader/output, BF16-to-NumPy and ensemble-CLI compatibility fixes.
FRCRN verification imports the surrounding MLNC checkout's unchanged FRCRN code.

The runner creates four complete Teacher logits arrays, preserving manifest order
and recording sample IDs and input/output hashes. Teachers are frozen Fold0 and
EMA BEATs; their input is FRCRN residual noise, single pass, without an added
external RMS operation, matching the team's new CED online helper. The Student
normalizes RMS internally. This protocol must not be confused with the older
MoE helper that normalizes Teacher input externally.

The Student uses shared Blocks 0–9, a spectral/token router, shared/low/high FFNs
in Blocks 10–11 and two classifier heads. It trains for 15 epochs, batch 64,
BF16, AdamW, weight decay 0.05, two-epoch warmup then cosine; backbone/expert/router
LRs are 4e−5/1.2e−4/2e−4. Preserve the team's CE/KD/router loss, temperature 2,
Mixup and label smoothing. Exact settings are in `config/team_kd_original_run.json`.

Outputs under `benchmark_results/team_kd_moe_full_20261102/` include `best.pt`,
`last.pt`, resumable state, history and final validation of Student, Student TTA
and the dual-Teacher ensemble separately. Test is not used. This experiment is
distinct from the nine **CE-only** baselines.

## Source provenance and scope

`config/source_manifest.json` records original and packaged file hashes and marks
setup/path portability edits. Training equations, architecture and metric
definitions are preserved. Dataset metadata are pinned in `config/hf_dataset_*`.
Third-party source snapshots retain their licenses; revisions are recorded by
`prepare_ce9.py`. `metadata/class_labels_indices.csv` is the small upstream
AudioSet class map required by EfficientAT, not an audio dataset or result file.

Only this `imp/mix_dataset_v2` directory is added. Existing MLNC code and local
uncommitted FRCRN changes are not included. No old 21-label or pre-clean mixture
experiments are added by this commit.
