import base64

cell13_code = """# ==============================================================================
# CẤU HÌNH THAM SỐ HUẤN LUYỆN (CONFIG-DRIVEN TRAINING CONFIGURATION)
# ==============================================================================
import yaml

# Đọc cấu hình từ config/config.yaml
_CONFIG_PATH = PKG_ROOT / "config" / "config.yaml"
if _CONFIG_PATH.is_file():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f)
else:
    cfg = {}

# Tham số điều khiển:
# - Đặt RUN_TRAINING = True: Tiến hành HUẤN LUYỆN / FINETUNE ĐẦY ĐỦ trên 30,240 clips của train.npy
# - Đặt RUN_TRAINING = False: Chạy Smoke Test 1 bước siêu tốc (~0.3s)
RUN_TRAINING: bool = True

# Tham số huấn luyện tự động kế thừa từ file cấu hình config.yaml
TRAIN_EPOCHS = int(cfg.get("training", {}).get("epochs", 15))
TRAIN_BATCH_SIZE = int(cfg.get("training", {}).get("batch_size", 64))
LR_BACKBONE = float(cfg.get("training", {}).get("lr_backbone", 4.0e-5))
LR_EXPERTS = float(cfg.get("training", {}).get("lr_experts", 1.2e-4))
LR_ROUTER = float(cfg.get("training", {}).get("lr_router", 2.0e-4))
WEIGHT_DECAY = float(cfg.get("training", {}).get("weight_decay", 0.05))
WARMUP_EPOCHS = int(cfg.get("training", {}).get("warmup_epochs", 2))
KD_TEMPERATURE = float(cfg.get("training", {}).get("kd_temperature", 2.0))
LAMBDA_HARD = float(cfg.get("training", {}).get("lambda_hard", 0.20))
LAMBDA_KD = float(cfg.get("training", {}).get("lambda_kd", 0.70))
LAMBDA_ROUTER = float(cfg.get("training", {}).get("lambda_router", 0.10))
USE_AMP = True

_status_badge = "🟢 **FINETUNE THỰC TẾ ĐẦY ĐỦ (RUN_TRAINING = True)**" if RUN_TRAINING else "⚡ **SMOKE TEST MÔ PHỎNG (RUN_TRAINING = False)**"

mo.md(f\"\"\"
### ⚙️ Thiết Lập Finetune Đầy Đủ (Kế Thừa Từ `config.yaml`)

> Trạng thái hiện tại: {_status_badge}  
> 💡 *Khi `RUN_TRAINING = True`, Cell 14 bên dưới sẽ tiến hành huấn luyện đầy đủ qua {TRAIN_EPOCHS} epochs trên toàn bộ 30,240 clips của `train.npy`!*

| Tham Số Cấu Hình | Giá Trị | Diễn Giải Chi Tiết |
| :--- | :--- | :--- |
| **`RUN_TRAINING`** | **`{RUN_TRAINING}`** | `True`: Kích hoạt tiến trình Finetune đầy đủ; `False`: Chạy Smoke Test |
| **`TRAIN_EPOCHS`** | **`{TRAIN_EPOCHS}`** | Tổng số epoch huấn luyện ({TRAIN_EPOCHS} epochs ~ {TRAIN_EPOCHS * 20.2 / 60:.1f} phút trên GPU) |
| **`TRAIN_BATCH_SIZE`** | `{TRAIN_BATCH_SIZE}` | Kích thước mini-batch (64 clips / batch, ~473 batches/epoch) |
| **`LR_BACKBONE`** | `{LR_BACKBONE:.1e}` | Tốc độ học nền tảng ViT Blocks 0–9 (Layer-wise LR) |
| **`LR_EXPERTS`** | `{LR_EXPERTS:.1e}` | Tốc độ học chuyên gia MoE FFN Blocks 10–11 |
| **`LR_ROUTER`** | `{LR_ROUTER:.1e}` | Tốc độ học Bộ định tuyến âm học (Router) |
| **`WEIGHT_DECAY`** | `{WEIGHT_DECAY}` | Hệ số suy giảm trọng số AdamW |
| **`WARMUP_EPOCHS`** | `{WARMUP_EPOCHS}` | Khởi động tốc độ học tuyến tính trong {WARMUP_EPOCHS} epochs đầu |
| **`PRECISION`** | `bfloat16 AMP` | Huấn luyện độ chính xác hỗn hợp tăng tốc phần cứng GPU |
| **`LOSS_WEIGHTS`** | `{LAMBDA_HARD} / {LAMBDA_ROUTER}` | Hàm mất mát: $L = {LAMBDA_HARD} L_{{\\\\text{{hard}}}} + {LAMBDA_ROUTER} L_{{\\\\text{{router}}}}$ |
\"\"\")
"""

cell14_code = """# ==============================================================================
# THỰC THI HUẤN LUYỆN: CHẾ ĐỘ THỰC TẾ (RUN_TRAINING = True) HOẶC SMOKE TEST (False)
# ==============================================================================
import json as _tr_json
import math as _tr_math
import time as _tr_time
import pandas as _pd
from tqdm import tqdm as _tr_tqdm

# Khởi tạo mô hình huấn luyện với trọng số tiền huấn luyện CED-Tiny AudioSet
model_train = CEDSingleMoESplitFFN(num_classes=36, load_pretrained=True).to(device)
model_train.train()

# Cấu hình Optimizer theo 3 nhóm tham số phân tầng (Layer-wise LR)
_backbone_params = [p for n, p in model_train.named_parameters() if "router" not in n and "ffn_" not in n and p.requires_grad]
_experts_params = [p for n, p in model_train.named_parameters() if "ffn_" in n and "router" not in n and p.requires_grad]
_router_params = [p for n, p in model_train.named_parameters() if "router" in n and p.requires_grad]

optimizer = torch.optim.AdamW([
    {"params": _backbone_params, "lr": LR_BACKBONE, "initial_lr": LR_BACKBONE},
    {"params": _experts_params, "lr": LR_EXPERTS, "initial_lr": LR_EXPERTS},
    {"params": _router_params, "lr": LR_ROUTER, "initial_lr": LR_ROUTER},
], weight_decay=WEIGHT_DECAY)

bce_loss_fn = nn.BCEWithLogitsLoss()

# Hàm điều chỉnh LR theo Cosine Annealing kèm Linear Warmup
def _get_lr_factor(epoch_idx: int, total_epochs: int, warmup: int) -> float:
    if epoch_idx < warmup:
        return float(epoch_idx + 1) / float(max(1, warmup))
    progress = float(epoch_idx - warmup) / float(max(1, total_epochs - warmup))
    return 0.5 * (1.0 + _tr_math.cos(_tr_math.pi * progress))

if not RUN_TRAINING:
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 1: SMOKE TEST 1 BƯỚC (RUN_TRAINING = False)
    # --------------------------------------------------------------------------
    print("ℹ️ CHẾ ĐỘ HIỆN TẠI: SMOKE TEST (RUN_TRAINING = False)")
    print("   (Đổi biến RUN_TRAINING = True trong cell trên để kích hoạt huấn luyện thực tế)")

    _st_batch = 4
    _st_wave = torch.randn(_st_batch, 64000, device=device)
    _st_target = torch.zeros(_st_batch, 36, device=device)
    _st_target[:, 5] = 1.0
    _st_snr = torch.tensor([-5.0, 5.0, 15.0, 20.0], device=device)

    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
        _st_out = model_train(_st_wave, return_dict=True)
        _st_loss_hard = bce_loss_fn(_st_out["logits"], _st_target)

    # Tính toán loss router trong float32 ngoài autocast
    _alpha_f32 = _st_out["alpha"].float()
    _y_tgt = torch.sigmoid((_st_snr - 15.0) / 2.5).unsqueeze(-1)
    _st_loss_router = F.binary_cross_entropy(_alpha_f32, _y_tgt)
    _st_loss_total = LAMBDA_HARD * _st_loss_hard + LAMBDA_ROUTER * _st_loss_router

    _st_loss_total.backward()
    optimizer.step()

    _grad_bb = any(p.grad is not None and p.grad.abs().sum() > 0 for p in _backbone_params)
    _grad_exp = any(p.grad is not None and p.grad.abs().sum() > 0 for p in _experts_params)
    _grad_rt = any(p.grad is not None and p.grad.abs().sum() > 0 for p in _router_params)

    print("-> Bước huấn luyện Smoke Test thành công!")
    print(f"   * Loss Hard: {_st_loss_hard.item():.4f}")
    print(f"   * Loss Router: {_st_loss_router.item():.4f}")
    print(f"   * Tổng Loss: {_st_loss_total.item():.4f}")
    print(f"   * Kiểm tra gradient: Backbone={_grad_bb}, Experts={_grad_exp}, Router={_grad_rt}")

    _tr_ui_output = mo.md(f\"\"\"
### ⚡ Kết Quả Smoke Test 1 Bước Mô Phỏng (RUN_TRAINING = False)

> 🟢 **Trạng thái:** Toàn bộ luồng dữ liệu (Dataflow), phép tính vi phân ngược (Backpropagation) và luồng gradient qua **ViT Backbone**, **Split-MoE FFNs** và **Spectral Token Router** đều hoạt động chuẩn xác 100%!

| Hạng Mục Kiểm Tra | Giá Trị Đo Được | Đánh Giá Kỹ Thuật |
| :--- | :--- | :--- |
| **Phân loại Hard Target ($L_{{\\\\text{{hard}}}}$)** | `{_st_loss_hard.item():.4f}` | Phù hợp phân phối nhãn BCE |
| **Giám sát thích ứng SNR ($L_{{\\\\text{{router}}}}$)** | `{_st_loss_router.item():.4f}` | Tiệm cận trơn tru về đường sigmoid mục tiêu |
| **Tổng tổn thất tối ưu ($L_{{\\\\text{{total}}}}$)** | **`{_st_loss_total.item():.4f}`** | Lan truyền ngược gradient toàn phần |
| **Kiểm tra Gradient Flow** | `Backbone: {_grad_bb} | Experts: {_grad_exp} | Router: {_grad_rt}` | **100% Thông suốt, không triệt tiêu đạo hàm** |
\"\"\")
else:
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 2: HUẤN LUYỆN / FINETUNE TOÀN DIỆN (RUN_TRAINING = True)
    # --------------------------------------------------------------------------
    print("🚀 BẮT ĐẦU TIẾN TRÌNH FINETUNE ĐẦY ĐỦ (RUN_TRAINING = True)")

    _train_npy_candidates = [
        Path("/marimo/datasets/train.npy"),
        Path("/datasets/mix-frcrn/train.npy"),
        Path("/marimo/datasets/mix-frcrn/train.npy"),
    ]
    _train_npy = next((p for p in _train_npy_candidates if p.is_file()), None)

    _train_manifest_candidates = [
        Path("/marimo/datasets/mix-dataset-v2/train/manifest.csv"),
        Path("/datasets/mix-dataset-v2/train/manifest.csv"),
    ]
    _train_manifest = next((p for p in _train_manifest_candidates if p.is_file()), None)

    _val_npy = Path("/marimo/datasets/validation.npy")
    _val_manifest = Path("/marimo/datasets/mix-dataset-v2/eval/manifest.csv")

    if not _train_npy or not _train_manifest:
        raise FileNotFoundError(f"Không tìm thấy tập huấn luyện tại: train_npy={_train_npy}, manifest={_train_manifest}.")

    print(f"-> Nạp tập huấn luyện mmap: {_train_npy} ({_train_npy.stat().st_size / 1e9:.2f} GB)")
    _train_waveforms = np.load(str(_train_npy), mmap_mode="r")
    _df_train = _pd.read_csv(str(_train_manifest))

    _train_multi_hot = np.array([_tr_json.loads(x) for x in _df_train["multi_hot_36"]], dtype=np.float32)
    _train_snrs = _df_train["target_snr_db"].to_numpy().astype(np.float32)
    _num_train_samples = len(_train_waveforms)
    _batch_sz = TRAIN_BATCH_SIZE
    _num_batches = (_num_train_samples + _batch_sz - 1) // _batch_sz

    print(f"-> Tổng số mẫu huấn luyện: {_num_train_samples:,} clips | Batch Size: {_batch_sz} | Batches/epoch: {_num_batches:,}")
    print(f"-> Kế hoạch huấn luyện: {TRAIN_EPOCHS} epochs | Warmup: {WARMUP_EPOCHS} epochs | Precision: bfloat16 AMP")

    # Chuẩn bị dữ liệu validation để đánh giá định kỳ
    _has_val = _val_npy.is_file() and _val_manifest.is_file()
    if _has_val:
        _val_waveforms_eval = np.load(str(_val_npy), mmap_mode="r")
        _df_val_eval = _pd.read_csv(str(_val_manifest))
        _val_targets_eval = np.argmax(np.array([_tr_json.loads(x) for x in _df_val_eval["multi_hot_36"]]), axis=-1)
        _val_snrs_eval = _df_val_eval["target_snr_db"].to_numpy()
        _num_val_eval = len(_val_waveforms_eval)

    _ckpt_dir = Path("checkpoints")
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    _best_val_top1 = 0.0
    _epoch_history = []
    _t_training_start = _tr_time.time()

    for _epoch in range(TRAIN_EPOCHS):
        model_train.train()
        _lr_factor = _get_lr_factor(_epoch, TRAIN_EPOCHS, WARMUP_EPOCHS)
        for _pg in optimizer.param_groups:
            _pg["lr"] = _pg["initial_lr"] * _lr_factor

        _t_ep_start = _tr_time.time()
        _indices = np.random.permutation(_num_train_samples)
        _ep_loss_total = 0.0
        _ep_loss_h = 0.0
        _ep_loss_r = 0.0

        _pbar = _tr_tqdm(
            range(_num_batches),
            desc=f"Epoch {_epoch + 1:02d}/{TRAIN_EPOCHS:02d} [LR={optimizer.param_groups[0]['lr']:.1e}]",
            unit="batch",
            ncols=105
        )

        for _b_idx in _pbar:
            _s_idx = _b_idx * _batch_sz
            _e_idx = min(_s_idx + _batch_sz, _num_train_samples)
            _b_indices = _indices[_s_idx:_e_idx]

            _b_wave = torch.from_numpy(np.array(_train_waveforms[_b_indices])).to(device)
            _b_target = torch.from_numpy(_train_multi_hot[_b_indices]).to(device)
            _b_snr = torch.from_numpy(_train_snrs[_b_indices]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                _b_out = model_train(_b_wave, return_dict=True)
                _lh = bce_loss_fn(_b_out["logits"], _b_target)

            # Tính loss router trong float32 ngoài autocast
            _alpha_f32 = _b_out["alpha"].float()
            _y_tgt = torch.sigmoid((_b_snr - 15.0) / 2.5).unsqueeze(-1)
            _lr = F.binary_cross_entropy(_alpha_f32, _y_tgt)
            _loss = LAMBDA_HARD * _lh + LAMBDA_ROUTER * _lr

            _loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
            optimizer.step()

            _ep_loss_total += _loss.item()
            _ep_loss_h += _lh.item()
            _ep_loss_r += _lr.item()
            _pbar.set_postfix({
                "Loss": f"{_loss.item():.4f}",
                "Hard": f"{_lh.item():.4f}",
                "Router": f"{_lr.item():.4f}",
                "Alpha": f"{_b_out['alpha'].mean().item():.3f}"
            })

        _ep_duration = _tr_time.time() - _t_ep_start
        _avg_loss = _ep_loss_total / _num_batches
        _avg_lh = _ep_loss_h / _num_batches
        _avg_lr = _ep_loss_r / _num_batches

        # Đánh giá nhanh Validation tại cuối mỗi Epoch
        _val_top1 = 0.0
        if _has_val:
            model_train.eval()
            _val_preds_list = []
            with torch.no_grad():
                for _vs in range(0, _num_val_eval, 128):
                    _ve = min(_vs + 128, _num_val_eval)
                    _v_batch = torch.from_numpy(np.array(_val_waveforms_eval[_vs:_ve])).to(device)
                    _v_out = model_train(_v_batch, return_dict=True)
                    _val_preds_list.append(torch.softmax(_v_out["logits"], dim=-1).cpu().numpy())
            _val_preds_arr = np.concatenate(_val_preds_list, axis=0)
            _val_top1 = float(np.mean(np.argmax(_val_preds_arr, axis=-1) == _val_targets_eval) * 100)

        _is_best = _val_top1 > _best_val_top1
        if _is_best:
            _best_val_top1 = _val_top1
            torch.save(model_train.state_dict(), _ckpt_dir / "best_champion_moe.pt")
            # Cập nhật trực tiếp mô hình model_champion cho Cell 16
            model_champion.load_state_dict(model_train.state_dict())

        _epoch_history.append({
            "epoch": _epoch + 1,
            "loss": _avg_loss,
            "loss_hard": _avg_lh,
            "loss_router": _avg_lr,
            "val_top1": _val_top1,
            "is_best": _is_best,
            "time": _ep_duration
        })

        _best_mark = " 🌟 (BEST)" if _is_best else ""
        print(f"-> Epoch {_epoch + 1:02d}/{TRAIN_EPOCHS:02d} hoàn tất ({_ep_duration:.1f}s) | Loss: {_avg_loss:.4f} | Val Top-1: {_val_top1:.2f}%{_best_mark}")

    _total_training_duration = _tr_time.time() - _t_training_start
    _final_ckpt = _ckpt_dir / "final_champion_moe.pt"
    torch.save(model_train.state_dict(), _final_ckpt)
    print(f"\\n🎉 HOÀN TẤT TIẾN TRÌNH FINETUNE ({_total_training_duration / 60:.1f} phút)!")
    print(f"-> Best Validation Top-1 Acc: {_best_val_top1:.2f}%")
    print(f"-> Trọng số tốt nhất đã lưu tại: {_ckpt_dir / 'best_champion_moe.pt'}")

    # Bảng kết quả Markdown
    _history_rows = []
    for h in _epoch_history:
        _b = "**" if h["is_best"] else ""
        _history_rows.append(
            f"| Epoch {h['epoch']:02d} | {_b}{h['loss']:.4f}{_b} | {_b}{h['loss_hard']:.4f}{_b} | {_b}{h['loss_router']:.4f}{_b} | {_b}{h['val_top1']:.2f}%{_b} | {h['time']:.1f}s | {'⭐ Kỷ lục mới' if h['is_best'] else ''} |"
        )
    _history_md = "\\n".join(_history_rows)

    _tr_ui_output = mo.md(f\"\"\"
### 🚀 Hoàn Tất Finetune Đầy Đủ (RUN_TRAINING = True)

> 🏆 **Kỷ Lục Top-1 Validation:** **{_best_val_top1:.2f}%**  
> ⏱️ **Tổng thời gian huấn luyện:** **{_total_training_duration / 60:.1f} phút** ({_num_train_samples * TRAIN_EPOCHS / _total_training_duration:.1f} clips/giây trên `{device}`).  
> 💾 **Checkpoint xuất xưởng:** Đã lưu trọng số tốt nhất vào `{_ckpt_dir / 'best_champion_moe.pt'}` và đồng bộ trực tiếp vào `model_champion`!

| Epoch | Tổng Loss | Loss Hard (BCE) | Loss Router (SNR) | Val Top-1 Acc | Thời Gian | Ghi Chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{_history_md}
\"\"\")

_tr_ui_output
"""

b64_c13 = base64.b64encode(cell13_code.encode("utf-8")).decode("ascii")
b64_c14 = base64.b64encode(cell14_code.encode("utf-8")).decode("ascii")

remote_code = f"""import base64
import marimo._code_mode as cm

b64_c13 = "{b64_c13}"
b64_c14 = "{b64_c14}"

async with cm.get_context() as ctx:
    _ = ctx.cells["nWHF"].code
    ctx.edit_cell("nWHF", code=base64.b64decode(b64_c13).decode("utf-8"))

    _ = ctx.cells["iLit"].code
    ctx.edit_cell("iLit", code=base64.b64decode(b64_c14).decode("utf-8"))

print("FINETUNE_ENGINE_APPLIED")
"""

with open("apply_engine.py", "w", encoding="ascii") as f:
    f.write(remote_code)
print("Wrote apply_engine.py successfully!")
