import marimo._code_mode as cm

code_train_50 = """import math
import time
import json

# 1. Cau hinh huan luyen lay truc tiep tu config.yaml (50 Epochs, Patience 8)
EPOCHS = cfg.get("training", {}).get("epochs", 50)
BATCH_SIZE = cfg.get("training", {}).get("batch_size", 32)
PATIENCE = cfg.get("training", {}).get("patience", 8)
WARMUP_RATIO = cfg.get("training", {}).get("warmup_ratio", 0.05)
WARMUP_EPOCHS = max(1, int(EPOCHS * WARMUP_RATIO))

LR_BACKBONE = 5.0e-5
LR_EXPERTS = 1.5e-4
LR_ROUTER = 2.0e-4

# Tao thu muc luu tru
Path("/marimo/checkpoints").mkdir(parents=True, exist_ok=True)
Path("/marimo/logs").mkdir(parents=True, exist_ok=True)

# num_workers=0 dam bao on dinh 100% trong kernel marimo
full_train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
full_val_loader = DataLoader(ds_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

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

criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

def get_lr_scale(ep: int, total_ep: int, warmup_ep: int) -> float:
    if ep < warmup_ep:
        return float(ep + 1) / float(max(1, warmup_ep))
    prog = float(ep - warmup_ep) / float(max(1, total_ep - warmup_ep))
    return 0.5 * (1.0 + math.cos(math.pi * prog))

print(f"\\U0001f680 BAT DAU HUAN LUYEN THEO CONFIG.YAML ({EPOCHS} Epochs, Batch Size={BATCH_SIZE}, Patience={PATIENCE})...")

training_history = []
best_top1 = 0.0
patience_counter = 0
final_breakdown = {}
final_metrics = {}

for epoch in range(1, EPOCHS + 1):
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
        torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 1.0)
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
        patience_counter = 0
        torch.save(moe_model.state_dict(), "/marimo/checkpoints/best_model_50epochs.pt")
        is_best_str = " * [BEST MODEL]"
    else:
        patience_counter += 1
        is_best_str = f" (Patience: {patience_counter}/{PATIENCE})"

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} ({elapsed:.1f}s) - Val Top-1: {top1_val:.2f}% | Top-3: {val_m9['top3']*100:.2f}% | BalAcc: {val_m9['balanced_acc']*100:.2f}% | mAP: {val_m9['map']:.4f} | AUC: {val_m9['auc']:.4f}{is_best_str}")

    epoch_record = {
        "epoch": epoch,
        "top1": top1_val,
        "metrics": val_m9,
        "snr_breakdown": val_snr_breakdown,
        "time_s": elapsed
    }
    training_history.append(epoch_record)
    final_breakdown = val_snr_breakdown
    final_metrics = val_m9

    # Luu log json sau moi epoch
    with open("/marimo/logs/summary_50epochs.json", "w", encoding="utf-8") as _f:
        json.dump({"best_top1": best_top1, "history": training_history}, _f, indent=2)

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\\U000026a0 Early stopping kich hoat tai Epoch {epoch} sau {PATIENCE} epochs khong cai thien.")
        break

# Markdown table hien thi ket qua tong hop
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
### \\U0001f3c6 [3] Ket Qua Huan Luyen CED-SplitFFN-MoE (Theo config.yaml: {EPOCHS} Epochs, Patience {PATIENCE})
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

async def update():
    async with cm.get_context() as ctx:
        _ = ctx.cells["DFbu"].code
        ctx.edit_cell("DFbu", code=code_train_50, hide_code=False)

await update()
print("DFbu updated to 50 epochs successfully!")
