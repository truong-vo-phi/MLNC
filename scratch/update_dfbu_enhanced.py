import marimo._code_mode as cm

code_train_enhanced = '''import math
import time
import json

# 1. Cau hinh huan luyen cai tien: Aux Head Supervision + Mixup + 3-Way Temporal TTA
EPOCHS = cfg.get("training", {}).get("epochs", 50)
BATCH_SIZE = cfg.get("training", {}).get("batch_size", 64)
PATIENCE = cfg.get("training", {}).get("patience", 8)
WARMUP_RATIO = cfg.get("training", {}).get("warmup_ratio", 0.05)
WARMUP_EPOCHS = max(1, int(EPOCHS * WARMUP_RATIO))

LR_BACKBONE = 4.0e-5
LR_EXPERTS = 1.2e-4
LR_ROUTER = 2.0e-4

Path("/marimo/checkpoints").mkdir(parents=True, exist_ok=True)
Path("/marimo/logs").mkdir(parents=True, exist_ok=True)

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

def get_lr_scale(ep: int, total_ep: int, warmup_ep: int) -> float:
    if ep < warmup_ep:
        return float(ep + 1) / float(max(1, warmup_ep))
    prog = float(ep - warmup_ep) / float(max(1, total_ep - warmup_ep))
    return 0.5 * (1.0 + math.cos(math.pi * prog))

print(f"[*] BAT DAU HUAN LUYEN CAI TIEN CED-SplitFFN-MoE (Aux Head Loss + Consistent Mixup + 3-Way TTA)...")

training_history = []
best_top1 = 0.0
best_epoch = 0
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
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=36).float()

        # Consistent Mixup (alpha=0.3, prob=0.7)
        if np.random.rand() < 0.7:
            b_sz = wav.size(0)
            perm = torch.randperm(b_sz, device=wav.device)
            lam = float(np.random.beta(0.3, 0.3))
            wav = lam * wav + (1.0 - lam) * wav[perm]
            target = lam * target + (1.0 - lam) * target[perm]
            snr_db = lam * snr_db + (1.0 - lam) * snr_db[perm]

        optimizer.zero_grad()
        out = moe_model(wav, return_dict=True)

        # 1. Fused classification loss
        loss_fused = -(target * F.log_softmax(out["logits"], dim=-1)).sum(dim=-1).mean()
        # 2. Auxiliary losses on separate heads (100% gradient directly into each head)
        loss_low = -(target * F.log_softmax(out["logits_low"], dim=-1)).sum(dim=-1).mean()
        loss_high = -(target * F.log_softmax(out["logits_high"], dim=-1)).sum(dim=-1).mean()
        # 3. Router supervision loss
        loss_router = moe_model.router.compute_router_loss(out["alpha"], snr_db)

        # Total multi-task loss
        loss = loss_fused + 0.3 * loss_low + 0.3 * loss_high + 0.15 * loss_router
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 1.0)
        optimizer.step()

        bs = wav.size(0)
        total_train_loss += loss.item() * bs
        preds = out["logits"].argmax(dim=-1)
        target_idx = target.argmax(dim=-1)
        correct_train += (preds == target_idx).sum().item()
        total_train_samples += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "fused": f"{loss_fused.item():.4f}",
            "acc": f"{(correct_train/total_train_samples)*100:.2f}%",
            "alpha": f"{out['alpha'].mean().item():.3f}"
        })

    # Evaluation with 3-Way Temporal TTA (+/-50ms) & Temperature Calibration (T=1.35)
    moe_model.eval()
    val_probs, val_targets, val_snrs, val_alphas = [], [], [], []
    shifts = [0, 800, -800] # +/-50ms at 16kHz
    
    with torch.no_grad():
        for batch in tqdm(full_val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Eval TTA]"):
            wav = batch["waveform"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            snr_db = batch["snr_db"].to(DEVICE)
            target_idx = target.argmax(dim=-1) if target.ndim == 2 else target

            logits_list = []
            for s in shifts:
                if s > 0:
                    w_s = F.pad(wav[:, s:], (0, s), mode="reflect")
                elif s < 0:
                    w_s = F.pad(wav[:, :-abs(s)], (abs(s), 0), mode="reflect")
                else:
                    w_s = wav
                out_s = moe_model(w_s, return_dict=True)
                logits_list.append(out_s["logits"])
            
            avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
            probs = F.softmax(avg_logits / 1.35, dim=-1)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(target_idx.cpu().numpy())
            val_snrs.append(snr_db.cpu().numpy())
            val_alphas.append(out_s["alpha"].cpu().numpy())

    all_p = np.concatenate(val_probs, axis=0)
    all_t = np.concatenate(val_targets, axis=0)
    all_s = np.concatenate(val_snrs, axis=0)
    all_a = np.concatenate(val_alphas, axis=0)

    cur_metrics = compute_9_metrics_dict(all_p, all_t, all_a)
    cur_snr_breakdown = compute_snr_breakdown(all_p, all_t, all_s)
    top1 = cur_metrics["top1"] * 100.0
    top3 = cur_metrics["top3"] * 100.0
    bal_acc = cur_metrics["balanced_acc"] * 100.0
    m_ap = cur_metrics["map"] * 100.0
    m_auc = cur_metrics["auc"] * 100.0

    epoch_dur = time.time() - t_start
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} ({epoch_dur:.1f}s) - Val TTA Top-1: {top1:.2f}% | Top-3: {top3:.2f}% | BalAcc: {bal_acc:.2f}% | mAP: {m_ap:.2f}% | AUC: {m_auc:.2f}% (Patience: {patience_counter}/{PATIENCE})")

    training_history.append({
        "epoch": epoch,
        "train_loss": total_train_loss / max(1, total_train_samples),
        "top1": top1,
        "metrics": cur_metrics,
        "snr_breakdown": cur_snr_breakdown,
        "duration_s": epoch_dur,
    })

    if top1 > best_top1:
        best_top1 = top1
        best_epoch = epoch
        patience_counter = 0
        final_breakdown = cur_snr_breakdown
        final_metrics = cur_metrics
        torch.save({
            "epoch": epoch,
            "model_state_dict": moe_model.state_dict(),
            "metrics": cur_metrics,
            "snr_breakdown": cur_snr_breakdown,
        }, "/marimo/checkpoints/best_model_enhanced.pt")
        print(f"[*] Checkpoint moi tot nhat tai Epoch {epoch} voi Val Top-1: {top1:.2f}%!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[*] Early stopping kich hoat tai Epoch {epoch} sau {PATIENCE} epochs khong cai thien.")
            break

summary_data = {
    "model_name": "CED-SingleMoE-SplitFFN-Enhanced",
    "best_epoch": best_epoch,
    "best_top1": best_top1,
    "final_metrics": final_metrics,
    "final_breakdown": final_breakdown,
    "history": training_history,
}
with open("/marimo/logs/summary_enhanced.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

print("[*] HUAN LUYEN VA NGHIEM THU HOAN TAT!")

_rows_md = [
    "| SNR Tier | Top-1 Acc | Top-3 Acc | mAP | AUC | Mean Alpha | Count |",
    "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
]
for _snr_k in ["-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"]:
    _d = final_breakdown.get(_snr_k, {})
    _rows_md.append(f"| **{_snr_k}** | {_d.get('top1', 0.0)*100:.2f}% | {_d.get('top3', 0.0)*100:.2f}% | {_d.get('map', 0.0):.4f} | {_d.get('auc', 0.0):.4f} | {_d.get('mean_alpha', 0.5):.3f} | {_d.get('count', 0)} |")

table_str = "\\n".join(_rows_md)
mo.md(
    f"""
### [NGHIEM THU] CED-SplitFFN-MoE (Enhanced Training)
- **Top-1 Accuracy TTA:** `{final_metrics.get('top1', 0.0)*100:.2f}%` (Epoch {best_epoch})
- **Top-3 Accuracy TTA:** `{final_metrics.get('top3', 0.0)*100:.2f}%`
- **Balanced Accuracy:** `{final_metrics.get('balanced_acc', 0.0)*100:.2f}%`
- **Macro-mAP:** `{final_metrics.get('map', 0.0):.4f}`
- **Macro-AUC:** `{final_metrics.get('auc', 0.0):.4f}`

#### Chi tiet hieu nang theo 6 dai SNR:
{table_str}
"""
)
'''

async def update_dfbu():
    async with cm.get_context() as ctx:
        print("Editing cell DFbu with Enhanced MoE training pipeline...")
        ctx.edit_cell("DFbu", code=code_train_enhanced, hide_code=False)
        print("Queueing cell DFbu for execution...")
        ctx.run_cell("DFbu")

await update_dfbu()
print("Cell DFbu updated and launched successfully!")
