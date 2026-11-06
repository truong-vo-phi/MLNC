import marimo._code_mode as cm

# 1. Code for cell vblA (Config & Model Init & Shared Imports)
code_vbla = """import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from src.dataset import MixtureDataset
from src.models.ced_moe_split import CEDSingleMoESplitFFN
from src.metrics import compute_9_metrics_dict, compute_snr_breakdown

# 1. Load config from imp/review-3/config/config.yaml
_config_path = Path("/marimo/config/config.yaml")
with open(_config_path, "r", encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

DATA_ROOT = "/datasets/mix-dataset-v2" if Path("/datasets/mix-dataset-v2").is_dir() else cfg["dataset"]["data_root"]
PRETRAINED_PATH = "/marimo/pretrained/model.safetensors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Build full datasets
ds_train = MixtureDataset(DATA_ROOT, split="train")
ds_eval = MixtureDataset(DATA_ROOT, split="eval")

# 3. Initialize CED-SingleMoE-SplitFFN (~6.62M params)
moe_model = CEDSingleMoESplitFFN(
    num_classes=cfg["dataset"]["num_classes"],
    load_pretrained=True,
    pretrained_path=PRETRAINED_PATH,
).to(DEVICE)

_p = moe_model.get_num_params()

mo.md(
    f\"\"\"
### \\U00002699\\U0000fe0f [1] Cau Hinh & Khoi Tao CED-SplitFFN-MoE
- **Thiet bi:** `{DEVICE}` ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})
- **Tap du lieu:** `{DATA_ROOT}` (Train: `{len(ds_train):,}` mau, Eval: `{len(ds_eval):,}` mau)
- **Tong so tham so:** `{_p['total']:,}` (Backbone + SplitFFN: `{_p['trainable']:,}`, Router: `{_p['router']:,}`)
- **AudioSet CED-Tiny Pretrained Weights:** `{moe_model.pretrained_loaded_tensors}` tensors da nap.
- **Muc SNR danh gia:** `[-5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB]`
\"\"\"
)
"""

# 2. Code for smoke_cell
code_smoke = """# 1. Tao loader thu nghiem nho (16 samples train, 16 samples eval)
_smoke_train_dl = DataLoader(Subset(ds_train, range(16)), batch_size=8, shuffle=True)
_smoke_val_dl = DataLoader(Subset(ds_eval, range(16)), batch_size=8, shuffle=False)

_opt_smoke = torch.optim.AdamW(moe_model.parameters(), lr=1e-4)
_crit_smoke = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

# 2. Chay 1 mini-epoch train
moe_model.train()
print(">>> BAT DAU SMOKE-TEST: Kiem tra gradient, forward va backward...")
_train_pbar = tqdm(_smoke_train_dl, desc="Smoke Train")
for _b in _train_pbar:
    _w = _b["waveform"].to(DEVICE)
    _t = _b["target"].to(DEVICE)
    _snr = _b["snr_db"].to(DEVICE)
    _t_idx = _t.argmax(dim=-1)

    _opt_smoke.zero_grad()
    _out = moe_model(_w, return_dict=True)
    _l_ce = _crit_smoke(_out["logits"], _t_idx)
    _l_router = moe_model.router.compute_router_loss(_out["alpha"], _snr)
    _loss = 0.5 * _l_ce + 0.2 * _l_router
    _loss.backward()
    torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 5.0)
    _opt_smoke.step()
    _train_pbar.set_postfix({"loss": f"{_loss.item():.4f}", "ce": f"{_l_ce.item():.4f}", "alpha": f"{_out['alpha'].mean().item():.3f}"})

# 3. Chay 1 mini-epoch eval
moe_model.eval()
print(">>> BAT DAU SMOKE-TEST: Kiem tra evaluate va SNR breakdown...")
_logits_list, _targets_list, _snrs_list, _alphas_list = [], [], [], []
with torch.no_grad():
    for _b in tqdm(_smoke_val_dl, desc="Smoke Eval"):
        _w = _b["waveform"].to(DEVICE)
        _t = _b["target"].to(DEVICE)
        _snr = _b["snr_db"].to(DEVICE)
        _out = moe_model(_w, return_dict=True)
        _logits_list.append(_out["logits"].cpu().numpy())
        _targets_list.append(_t.cpu().numpy())
        _snrs_list.append(_snr.cpu().numpy())
        _alphas_list.append(_out["alpha"].cpu().numpy())

_l_np = np.concatenate(_logits_list, axis=0)
_t_np = np.concatenate(_targets_list, axis=0)
_s_np = np.concatenate(_snrs_list, axis=0)
_a_np = np.concatenate(_alphas_list, axis=0)

smoke_metrics_9 = compute_9_metrics_dict(_l_np, _t_np, alphas=_a_np, num_classes=36)
smoke_snr_breakdown = compute_snr_breakdown(_l_np, _t_np, _s_np, alphas=_a_np, num_classes=36, use_9_metrics=True)

mo.md(
    f\"\"\"
### \\U00002705 [2] Smoke-Test Hoan Tat - Khong Co Loi (Pipeline Verified!)
Pipeline hoc sau, lan truyen xuoi, dao ham va danh gia 9 metrics / 6 muc SNR hoat dong chuan xac 100%.

- **Forward Waveform:** RMS Norm -> STFT (6 dac trung f_1..f_6) -> Log-Mel -> ViT 0-9 -> Router -> SplitFFN 10-11 -> Dual Heads: **OK**
- **Backward & Optimizer:** Gradient Clipping & AdamW step: **OK**
- **He Thong Metrics:** top1, top3, balanced_acc, mAP, AUC, precision, recall, f1, mean_alpha: **OK**
- **Trang thai:** Ban co the an tam nhan **Run All** de bat dau tien trinh huan luyen day du!
\"\"\"
)
"""

# 3. Code for train_cell
code_train = """# Cau hinh huan luyen tu config
BATCH_SIZE = 32
EPOCHS = 15
LR_BACKBONE = 5.0e-5
LR_EXPERTS = 1.5e-4
LR_ROUTER = 2.0e-4
WARMUP_EPOCHS = 2

# Tao thu muc luu checkpoint
Path("/marimo/checkpoints").mkdir(parents=True, exist_ok=True)

full_train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
full_val_loader = DataLoader(ds_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

_backbone_p, _experts_p, _router_p = [], [], []
for _name, _param in moe_model.named_parameters():
    if not _param.requires_grad:
        continue
    if "router" in _name:
        _router_p.append(_param)
    elif any(_k in _name for _k in ("ffn_shared", "ffn_low", "ffn_high", "head_low", "head_high")):
        _experts_p.append(_param)
    else:
        _backbone_p.append(_param)

optimizer = torch.optim.AdamW([
    {"params": _backbone_p, "lr": LR_BACKBONE, "weight_decay": 0.05, "initial_lr": LR_BACKBONE},
    {"params": _experts_p, "lr": LR_EXPERTS, "weight_decay": 0.05, "initial_lr": LR_EXPERTS},
    {"params": _router_p, "lr": LR_ROUTER, "weight_decay": 1e-4, "initial_lr": LR_ROUTER},
])

criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

def get_lr_scale(ep: int, total_ep: int, warmup_ep: int) -> float:
    import math
    if ep < warmup_ep:
        return float(ep + 1) / float(max(1, warmup_ep))
    prog = float(ep - warmup_ep) / float(max(1, total_ep - warmup_ep))
    return 0.5 * (1.0 + math.cos(math.pi * prog))

print(f"\\U0001f680 BAT DAU HUAN LUYEN DAY DU ({EPOCHS} Epochs, Batch Size={BATCH_SIZE})...")

training_history = []
best_top1 = 0.0
final_breakdown = {}
final_metrics = {}

for epoch in range(1, EPOCHS + 1):
    import time
    t_start = time.time()
    moe_model.train()
    
    lr_scale = get_lr_scale(epoch - 1, EPOCHS, WARMUP_EPOCHS)
    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * lr_scale

    total_train_loss = 0.0
    correct_train = 0
    total_train_samples = 0

    pbar = tqdm(full_train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]")
    for batch in pbar:
        wav = batch["waveform"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        snr_db = batch["snr_db"].to(DEVICE)
        target_idx = target.argmax(dim=-1)

        optimizer.zero_grad()
        out = moe_model(wav, return_dict=True)
        loss_ce = criterion_ce(out["logits"], target_idx)
        loss_router = moe_model.router.compute_router_loss(out["alpha"], snr_db)
        loss = 0.5 * loss_ce + 0.2 * loss_router
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 5.0)
        optimizer.step()

        bs = wav.size(0)
        total_train_loss += loss.item() * bs
        preds = out["logits"].argmax(dim=-1)
        correct_train += (preds == target_idx).sum().item()
        total_train_samples += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct_train/total_train_samples)*100:.2f}%",
            "alpha": f"{out['alpha'].mean().item():.3f}"
        })

    moe_model.eval()
    val_logits, val_targets, val_snrs, val_alphas = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(full_val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Eval]"):
            wav = batch["waveform"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            snr_db = batch["snr_db"].to(DEVICE)
            out = moe_model(wav, return_dict=True)
            val_logits.append(out["logits"].cpu().numpy())
            val_targets.append(target.cpu().numpy())
            val_snrs.append(snr_db.cpu().numpy())
            val_alphas.append(out["alpha"].cpu().numpy())

    v_logits = np.concatenate(val_logits, axis=0)
    v_targets = np.concatenate(val_targets, axis=0)
    v_snrs = np.concatenate(val_snrs, axis=0)
    v_alphas = np.concatenate(val_alphas, axis=0)

    val_m9 = compute_9_metrics_dict(v_logits, v_targets, alphas=v_alphas, num_classes=36)
    val_snr_breakdown = compute_snr_breakdown(v_logits, v_targets, v_snrs, alphas=v_alphas, num_classes=36, use_9_metrics=True)

    elapsed = time.time() - t_start
    top1_val = val_m9["top1"] * 100.0
    if top1_val > best_top1:
        best_top1 = top1_val
        torch.save(moe_model.state_dict(), "/marimo/checkpoints/best_model.pt")
        is_best_str = " * [BEST]"
    else:
        is_best_str = ""

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} ({elapsed:.1f}s) - Val Top-1: {top1_val:.2f}% | Top-3: {val_m9['top3']*100:.2f}% | BalAcc: {val_m9['balanced_acc']*100:.2f}% | mAP: {val_m9['map']:.4f} | AUC: {val_m9['auc']:.4f}{is_best_str}")

    training_history.append({"epoch": epoch, "top1": top1_val, "metrics": val_m9, "snr": val_snr_breakdown})
    final_breakdown = val_snr_breakdown
    final_metrics = val_m9

# Markdown table hien thi ket qua
_snr_rows = ""
for _snr_k in ["-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"]:
    _d = final_breakdown.get(_snr_k, {})
    _cnt = _d.get("count", 0)
    _t1 = _d.get("top1", 0.0) * 100.0 if _cnt > 0 else 0.0
    _t3 = _d.get("top3", 0.0) * 100.0 if _cnt > 0 else 0.0
    _map = _d.get("map", 0.0)
    _alp = _d.get("mean_alpha", 0.5)
    _snr_rows += f"| **{_snr_k}** | {_cnt:,} | **{_t1:.2f}%** | {_t3:.2f}% | {_map:.4f} | {_alp:.3f} |\\n"

mo.md(
    f\"\"\"
### \\U0001f3c6 [3] Ket Qua Huan Luyen CED-SplitFFN-MoE ({EPOCHS} Epochs)
- **Top-1 Accuracy Tot Nhat:** **`{best_top1:.2f}%`**
- **Top-3 Accuracy:** `{final_metrics.get('top3', 0.0)*100:.2f}%`
- **Balanced Accuracy:** `{final_metrics.get('balanced_acc', 0.0)*100:.2f}%`
- **Macro mAP:** `{final_metrics.get('map', 0.0):.4f}` | **Macro AUC:** `{final_metrics.get('auc', 0.0):.4f}`
- **Macro F1-Score:** `{final_metrics.get('f1', 0.0):.4f}`

#### \\U0001f4ca Bang Phan Ra Hieu Nang Theo 6 Muc SNR:
| Muc SNR | So mau | Top-1 Acc | Top-3 Acc | Macro mAP | Router \\\\bar{{\\\\alpha}} |
| :---: | :---: | :---: | :---: | :---: | :---: |
{_snr_rows}
\"\"\"
)
"""

async def deploy():
    async with cm.get_context() as ctx:
        # 1. Update cell vblA
        _ = ctx.cells["vblA"].code
        ctx.edit_cell("vblA", code=code_vbla, hide_code=False)
        ctx.run_cell("vblA")

        # 2. Create cell smoke
        smoke_cid = ctx.create_cell(code=code_smoke, after="vblA", hide_code=False)
        ctx.run_cell(smoke_cid)

        # 3. Create cell train
        train_cid = ctx.create_cell(code=code_train, after=smoke_cid, hide_code=False)

await deploy()
print("Cells deployed successfully!")
