import marimo._code_mode as cm

smoke_code = """# Smoke-test Kiem tra toan dien Loss cai tien (Aux Head Loss + Consistent Mixup + 3-Way TTA)
_smoke_train_dl = DataLoader(Subset(ds_train, range(16)), batch_size=8, shuffle=True)
_smoke_val_dl = DataLoader(Subset(ds_eval, range(16)), batch_size=8, shuffle=False)

_opt_smoke = torch.optim.AdamW(moe_model.parameters(), lr=1e-4)

# Test forward + Aux Head Loss + Mixup
moe_model.train()
print(">>> BAT DAU SMOKE-TEST: Kiem tra Aux Head Loss, Mixup va Backward...")
for _b in _smoke_train_dl:
    _w = _b["waveform"].to(DEVICE)
    _t = _b["target"].to(DEVICE)
    _snr = _b["snr_db"].to(DEVICE)
    if _t.ndim == 1:
        _t = F.one_hot(_t, num_classes=36).float()

    # Mixup
    _b_sz = _w.size(0)
    _perm = torch.randperm(_b_sz, device=_w.device)
    _lam = 0.7
    _w_m = _lam * _w + (1.0 - _lam) * _w[_perm]
    _t_m = _lam * _t + (1.0 - _t[_perm])
    _snr_m = _lam * _snr + (1.0 - _lam) * _snr[_perm]

    _opt_smoke.zero_grad()
    _out = moe_model(_w_m, return_dict=True)

    # Soft CrossEntropy
    _l_fused = -(_t_m * F.log_softmax(_out["logits"], dim=-1)).sum(dim=-1).mean()
    _l_low = -(_t_m * F.log_softmax(_out["logits_low"], dim=-1)).sum(dim=-1).mean()
    _l_high = -(_t_m * F.log_softmax(_out["logits_high"], dim=-1)).sum(dim=-1).mean()
    _l_router = moe_model.router.compute_router_loss(_out["alpha"], _snr_m)

    _loss = _l_fused + 0.3 * _l_low + 0.3 * _l_high + 0.15 * _l_router
    _loss.backward()
    torch.nn.utils.clip_grad_norm_(moe_model.parameters(), 1.0)
    _opt_smoke.step()
    print(f"  Step Smoke: loss={_loss.item():.4f}, fused={_l_fused.item():.4f}, low={_l_low.item():.4f}, high={_l_high.item():.4f}, alpha={_out['alpha'].mean().item():.3f}")

# Test 3-Way TTA Eval
moe_model.eval()
print(">>> BAT DAU SMOKE-TEST: Kiem tra 3-Way TTA (+/-50ms) va SNR breakdown...")
_probs_list, _targets_list, _snrs_list, _alphas_list = [], [], [], []
with torch.no_grad():
    for _b in _smoke_val_dl:
        _w = _b["waveform"].to(DEVICE)
        _t = _b["target"].to(DEVICE)
        _snr = _b["snr_db"].to(DEVICE)
        _t_idx = _t.argmax(dim=-1) if _t.ndim == 2 else _t

        _l_list = []
        for _s in [0, 800, -800]:
            if _s > 0:
                _ws = F.pad(_w[:, _s:], (0, _s), mode="reflect")
            elif _s < 0:
                _ws = F.pad(_w[:, :-abs(_s)], (abs(_s), 0), mode="reflect")
            else:
                _ws = _w
            _o = moe_model(_ws, return_dict=True)
            _l_list.append(_o["logits"])
        
        _avg_logits = torch.stack(_l_list, dim=0).mean(dim=0)
        _probs = F.softmax(_avg_logits / 1.35, dim=-1)
        _probs_list.append(_probs.cpu().numpy())
        _targets_list.append(_t_idx.cpu().numpy())
        _snrs_list.append(_snr.cpu().numpy())
        _alphas_list.append(_o["alpha"].cpu().numpy())

_p_all = np.concatenate(_probs_list, axis=0)
_t_all = np.concatenate(_targets_list, axis=0)
_s_all = np.concatenate(_snrs_list, axis=0)
_a_all = np.concatenate(_alphas_list, axis=0)

_m = compute_9_metrics_dict(_p_all, _t_all, _a_all)
_snr_res = compute_snr_breakdown(_p_all, _t_all, _s_all)
print(f"SMOKE-TEST PASSED: Top-1={_m['top1']*100:.2f}%, Mean Alpha={_m['mean_alpha']:.3f}")
mo.md(f"**Smoke-Test Cai Tien:** PASSED (Top-1: {_m['top1']*100:.2f}%, Alpha: {_m['mean_alpha']:.3f})")
"""

async def update_cell():
    async with cm.get_context() as ctx:
        print("Updating cell nrKG with enhanced smoke test...")
        ctx.edit_cell("nrKG", code=smoke_code, hide_code=False)
        print("Running cell nrKG...")
        ctx.run_cell("nrKG")

await update_cell()
print("Cell nrKG updated and executed successfully!")
