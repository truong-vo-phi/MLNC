import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

widget_cell_code = '''
# ==============================================================================
# BẢNG ĐIỀU KHIỂN TƯƠNG TÁC HUẤN LUYỆN (TRAINING DASHBOARD)
# ==============================================================================
run_training = mo.ui.switch(
    value=False,
    label="🚀 Kích hoạt Huấn Luyện Thực Tế (RUN_TRAINING = True)",
)
train_steps = mo.ui.slider(
    start=5,
    stop=50,
    step=5,
    value=10,
    label="Số bước huấn luyện (Steps)",
)
train_lr = mo.ui.dropdown(
    options=["5e-5", "1e-4", "2e-4"],
    value="1e-4",
    label="Tốc độ học (Learning Rate)",
)

mo.vstack([
    mo.md("""
### 🎛️ Bảng Điều Khiển Huấn Luyện Tương Tác (Training Controls)

> 💡 **Hướng dẫn cấu hình:**
> - **`RUN_TRAINING = False` (Mặc định):** Chạy Smoke Test 1 bước mô phỏng siêu tốc (~0.5s) để bảo toàn tốc độ khi bạn bấm **"Run All"**.
> - **`RUN_TRAINING = True` (Gạt công tắc sang phải):** Kích hoạt vòng lặp huấn luyện thực tế trên tập `train.npy` (30,240 clips) với số bước và tốc độ học bạn đã chọn bên dưới!
"""),
    mo.hstack([run_training, train_steps, train_lr], gap=2),
])
'''

hstk_new_code = '''
# ==============================================================================
# THỰC THI HUẤN LUYỆN: CHẾ ĐỘ THỰC TẾ (RUN_TRAINING = True) HOẶC SMOKE TEST (False)
# ==============================================================================
import json as _tr_json
import time as _tr_time
import pandas as _pd
from tqdm import tqdm as _tr_tqdm

model_train = CEDSingleMoESplitFFN(num_classes=36).to(device)
model_train.train()
optimizer = torch.optim.AdamW(
    model_train.parameters(), lr=float(train_lr.value), weight_decay=0.05
)
bce_loss_fn = nn.BCEWithLogitsLoss()

if not run_training.value:
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 1: SMOKE TEST 1 BƯỚC (RUN_TRAINING = False)
    # --------------------------------------------------------------------------
    print("ℹ️ CHẾ ĐỘ HIỆN TẠI: SMOKE TEST (RUN_TRAINING = False)")
    print("   (Gạt công tắc ở bảng điều khiển trên sang 'True' để chạy huấn luyện thực tế)")
    
    _st_batch = 4
    _st_wave = torch.randn(_st_batch, 64000, device=device)
    _st_target = torch.zeros(_st_batch, 36, device=device)
    _st_target[:, 5] = 1.0
    _st_snr = torch.tensor([-5.0, 5.0, 15.0, 20.0], device=device)
    _st_t0_logits = torch.randn(_st_batch, 36, device=device)
    _st_tema_logits = torch.randn(_st_batch, 36, device=device)

    optimizer.zero_grad()
    _st_out = model_train(_st_wave, return_dict=True)
    
    _st_loss_hard = bce_loss_fn(_st_out["logits"], _st_target)
    _T = 2.0
    _st_ytarget = model_train.router.compute_target_soft_label(_st_snr)
    _st_p_t0 = F.softmax(_st_t0_logits / _T, dim=-1)
    _st_p_tema = F.softmax(_st_tema_logits / _T, dim=-1)
    _st_p_teacher = (1.0 - _st_ytarget) * _st_p_t0 + _st_ytarget * _st_p_tema
    
    _st_loss_kd_global = F.kl_div(
        F.log_softmax(_st_out["logits"] / _T, dim=-1),
        _st_p_teacher,
        reduction="batchmean"
    ) * (_T ** 2)
    _st_loss_kd_low = F.kl_div(
        F.log_softmax(_st_out["logits_low"] / _T, dim=-1),
        _st_p_t0,
        reduction="batchmean"
    ) * (_T ** 2)
    _st_loss_kd_high = F.kl_div(
        F.log_softmax(_st_out["logits_high"] / _T, dim=-1),
        _st_p_tema,
        reduction="batchmean"
    ) * (_T ** 2)
    _st_loss_kd = _st_loss_kd_global + 0.5 * (_st_loss_kd_low + _st_loss_kd_high)
    _st_loss_router = model_train.router.compute_router_loss(_st_out["alpha"], _st_snr)
    
    _st_total_loss = 0.2 * _st_loss_hard + 0.7 * _st_loss_kd + 0.1 * _st_loss_router
    _st_total_loss.backward()
    optimizer.step()
    
    print("-> Bước huấn luyện Smoke Test thành công!")
    print(f"   * Loss Hard: {_st_loss_hard.item():.4f}")
    print(f"   * Loss KD Kép: {_st_loss_kd.item():.4f}")
    print(f"   * Loss Router: {_st_loss_router.item():.4f}")
    print(f"   * Tổng Loss: {_st_total_loss.item():.4f}")
    
    _tr_ui_output = mo.md(f"""
> 🟢 **Chế độ: Smoke Test (1 Step - Khởi tạo Gradient hợp lệ)**
> - **Loss Hard:** `{_st_loss_hard.item():.4f}` | **Loss KD Kép:** `{_st_loss_kd.item():.4f}` | **Loss Router:** `{_st_loss_router.item():.4f}` | **Tổng Loss:** `{_st_total_loss.item():.4f}`
> - *Để kích hoạt vòng lặp huấn luyện thực tế trên tập `train.npy` (30,240 clips), vui lòng gạt công tắc **`RUN_TRAINING`** ở trên sang **True**.*
""")
else:
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 2: HUẤN LUYỆN THỰC TẾ TRÊN TẬP DỮ LIỆU TRAIN.NPY (RUN_TRAINING = True)
    # --------------------------------------------------------------------------
    _num_steps = int(train_steps.value)
    _batch_sz = 16
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN THỰC TẾ: {_num_steps} STEPS | BATCH SIZE = {_batch_sz} | LR = {train_lr.value}")
    
    _train_npy_p = Path("/marimo/datasets/train.npy")
    _train_csv_p = Path("/marimo/datasets/mix-dataset-v2/train/manifest.csv")
    
    if not _train_npy_p.is_file() or not _train_csv_p.is_file():
        raise FileNotFoundError("Không tìm thấy train.npy hoặc train manifest.csv.")
        
    _train_waves = np.load(str(_train_npy_p), mmap_mode="r")
    _df_train = _pd.read_csv(str(_train_csv_p), nrows=_num_steps * _batch_sz)
    
    _tr_t0 = _tr_time.time()
    _loss_history = []
    
    _pbar_tr = _tr_tqdm(
        range(_num_steps),
        desc=f"Huấn luyện CED-SingleMoE ({_num_steps} steps)",
        unit="step"
    )
    for _step in _pbar_tr:
        _s_idx = _step * _batch_sz
        _e_idx = _s_idx + _batch_sz
        _batch_w = torch.from_numpy(np.array(_train_waves[_s_idx:_e_idx])).to(device)
        _batch_targets = torch.tensor(
            [_tr_json.loads(x) for x in _df_train["multi_hot_36"].iloc[_s_idx:_e_idx]],
            dtype=torch.float32,
            device=device
        )
        _batch_snrs = torch.tensor(
            _df_train["target_snr_db"].iloc[_s_idx:_e_idx].to_numpy(),
            dtype=torch.float32,
            device=device
        )
        
        optimizer.zero_grad()
        _out_tr = model_train(_batch_w, return_dict=True)
        
        _l_hard = bce_loss_fn(_out_tr["logits"], _batch_targets)
        _l_router = model_train.router.compute_router_loss(_out_tr["alpha"], _batch_snrs)
        _l_tot = _l_hard + 0.1 * _l_router
        
        _l_tot.backward()
        optimizer.step()
        
        _loss_history.append(_l_tot.item())
        _pbar_tr.set_postfix(
            loss=f"{_l_tot.item():.4f}",
            hard=f"{_l_hard.item():.4f}",
            router=f"{_l_router.item():.4f}"
        )
        
    _tr_duration = _tr_time.time() - _tr_t0
    print(f"\\n✓ Hoàn tất {_num_steps} bước huấn luyện trong {_tr_duration:.2f}s (~{_tr_duration/_num_steps:.2f}s/step)!")
    print(f"  Loss khởi điểm: {_loss_history[0]:.4f} -> Loss kết thúc: {_loss_history[-1]:.4f}")
    
    _tr_ui_output = mo.md(f"""
### 🏆 Kết Quả Huấn Luyện Thực Tế Thành Công!
> ⚡ **Thống kê:** Hoàn tất **{_num_steps} steps** ({_num_steps * _batch_sz:,} mẫu) trong **{_tr_duration:.2f} giây** trên thiết bị `{device}`.
> - **Loss khởi điểm:** `{_loss_history[0]:.4f}` ──> **Loss sau tối ưu:** `{_loss_history[-1]:.4f}` (Giảm `{(_loss_history[0] - _loss_history[-1]):.4f}`)
> - **Tốc độ:** `{_tr_duration / _num_steps:.2f} giây/step` (~`{_batch_sz / (_tr_duration / _num_steps):.1f} clips/giây`).
""")

_tr_ui_output
'''

deploy_script = """
import marimo._code_mode as cm

async def add_training_controls():
    async with cm.get_context() as ctx:
        # Check if training_control cell already exists
        existing_cid = None
        for cid, cell in ctx.cells.items():
            if "run_training" in cell.code and "mo.ui.switch" in cell.code:
                existing_cid = cid
                break
        
        if existing_cid:
            print("Editing existing control cell " + str(existing_cid) + "...")
            ctx.edit_cell(existing_cid, """ + repr(widget_cell_code.strip()) + """)
            ctrl_cid = existing_cid
        else:
            print("Creating new training control cell before Hstk...")
            ctrl_cid = ctx.create_cell(""" + repr(widget_cell_code.strip()) + """, before="Hstk", hide_code=False)
            
        print("Editing Hstk...")
        ctx.edit_cell("Hstk", """ + repr(hstk_new_code.strip()) + """)
        
        ctx.run_cell(ctrl_cid)
        ctx.run_cell("Hstk")

await add_training_controls()
"""

rc = execute_remote(deploy_script)
print(f"Deployment result code: {rc}")
