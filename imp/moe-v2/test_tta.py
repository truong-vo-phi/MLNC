import sys
sys.path.insert(0, "/marimo/moe-v2")
import torch, numpy as np, torch.nn.functional as F
from src.dataset import Mixture36Dataset, NoiseCacheDataset, build_dataloader
from src.models import CEDHydraLoRAMoE, CEDSingleMoESplitFFN
from src.metrics import compute_snr_breakdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_base = Mixture36Dataset(root="/marimo/datasets/mix-dataset-v2", split="eval")
val_cache = NoiseCacheDataset(val_base, "/marimo/datasets/validation.npy")
val_loader = build_dataloader(val_cache, batch_size=64, shuffle=False, num_workers=0)

model = CEDHydraLoRAMoE(num_classes=36).to(device)
ckpt = torch.load("/marimo/moe-v2/checkpoints/ced-hydralora-moe/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

def tta_forward(m, wave, shift=0):
    if shift != 0:
        if shift > 0:
            w = F.pad(wave[:, shift:], (0, shift), mode="reflect")
        else:
            s = abs(shift)
            w = F.pad(wave[:, :-s], (s, 0), mode="reflect")
    else:
        w = wave
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(w)
        logits = out["logits"] if isinstance(out, dict) else out
    return logits.float()

all_probs_direct, all_probs_tta, all_targets, all_snrs = [], [], [], []
with torch.no_grad():
    for b in val_loader:
        w = b["waveform"].to(device)
        t = b["target_idx"].numpy()
        s = b["snr_db"].numpy()
        
        # Direct
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out0 = model(w)
            l0 = out0["logits"] if isinstance(out0, dict) else out0
            p_dir = F.softmax(l0.float(), dim=-1).cpu().numpy()
        
        # TTA (+/- 800 samples = 50ms)
        lp = tta_forward(model, w, shift=800)
        ln = tta_forward(model, w, shift=-800)
        l_tta = (l0.float() + 0.5 * lp + 0.5 * ln) / 2.0
        p_tta = F.softmax(l_tta / 1.35, dim=-1).cpu().numpy()
        
        all_probs_direct.append(p_dir)
        all_probs_tta.append(p_tta)
        all_targets.append(t)
        all_snrs.append(s)

targets = np.concatenate(all_targets)
snrs = np.concatenate(all_snrs)
res_dir = compute_snr_breakdown(np.concatenate(all_probs_direct), targets, snrs)
res_tta = compute_snr_breakdown(np.concatenate(all_probs_tta), targets, snrs)

print(f"DIRECT: Top-1={res_dir['top1_accuracy']*100:.2f}%, Top-3={res_dir['top3_accuracy']*100:.2f}%, mAP={res_dir['mAP_macro']*100:.2f}%, AUC={res_dir['macro_auc']*100:.2f}%")
print(f"TTA:    Top-1={res_tta['top1_accuracy']*100:.2f}%, Top-3={res_tta['top3_accuracy']*100:.2f}%, mAP={res_tta['mAP_macro']*100:.2f}%, AUC={res_tta['macro_auc']*100:.2f}%")
for snr in [-5, 0, 5, 10, 15, 20]:
    print(f"  SNR {snr:2d} dB: Direct={res_dir.get(f'acc_snr_{snr}',0)*100:.2f}%, TTA={res_tta.get(f'acc_snr_{snr}',0)*100:.2f}%")
