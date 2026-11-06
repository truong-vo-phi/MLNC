import base64
from pathlib import Path

# 1. New IBDh (architecture using repo BEATClassifier)
new_ibdh = r'''# Kiến trúc BEATs Audio Transformer từ repo MLNC
import sys
import yaml

if "/marimo" not in sys.path:
    sys.path.insert(0, "/marimo")

from src.model.beat import BEATClassifier
from src.model.beat.beat import BEATConfig

with open("/marimo/config/beat.yaml", encoding="utf-8") as _f:
    _beat_raw = yaml.safe_load(_f)

def build_beats_model(num_classes: int = 36) -> BEATClassifier:
    cfg = BEATConfig.from_dict({**_beat_raw["model"], "num_classes": num_classes})
    return BEATClassifier(cfg, _beat_raw.get("frontend"))
'''

# 2. New HcAs (instantiation & verification)
new_hcas = r'''# Khởi tạo mô hình Teacher (đóng băng) và Student (huấn luyện)
teacher_model = build_beats_model(num_classes=36).to(device)
student_model = build_beats_model(num_classes=36).to(device)

_beats_ckpt_path = Path("/marimo/pretrained/BEATs_iter3_plus_AS2M.pt")
if _beats_ckpt_path.is_file():
    teacher_model.load_pretrained(str(_beats_ckpt_path))
    student_model.load_pretrained(str(_beats_ckpt_path))
    print(f"✓ Đã nạp thành công trọng số khởi tạo từ {_beats_ckpt_path.name}!")

teacher_model.eval()
for _p in teacher_model.parameters():
    _p.requires_grad = False

_t_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
_s_trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6
_s_frozen = sum(p.numel() for p in student_model.parameters() if not p.requires_grad) / 1e6

# Kiểm tra 1 forward pass trên batch giả lập
_dummy_x = torch.randn(2, 64000, device=device)
_dummy_y = torch.randint(0, 36, (2,), device=device)
with torch.no_grad():
    _t_out = teacher_model(_dummy_x).logits
_s_out = student_model(_dummy_x).logits

_crit = BornAgainLogitLoss(temperature=2.0, kd_weight=0.8)
_test_loss, _test_terms = _crit(_s_out, _t_out, _dummy_y)
_test_loss.backward()
_grad_ok = student_model.head.fc.weight.grad is not None
student_model.zero_grad(set_to_none=True)

mo.md(f"""
### 🧪 Kiểm tra Kiến trúc & Forward-Backward
- **Teacher BEATs**: {_t_params:.2f}M tham số (**100% Đóng băng**).
- **Student BEATs**: {_s_trainable:.2f}M tham số huấn luyện | {_s_frozen:.2f}M đóng băng.
- Kích thước Logits: Teacher `{tuple(_t_out.shape)}` | Student `{tuple(_s_out.shape)}`.
- Kiểm tra Gradient: `{'PASS ✓' if _grad_ok else 'FAIL ✗'}` (Gradient chỉ truyền vào Student).
- Loss test batch: `{_test_terms}`.
""")
'''

# 3. New vOZk (eval function using .logits)
new_vozk = r'''# Hàm Đánh giá Toàn diện theo từng Mốc SNR (-5, 0, 5, 10, 15, 20 dB)
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_preds, all_targets, all_snrs = [], [], []

    for batch in loader:
        wavs = batch["waveform"].to(device)
        targets = batch["target"].to(device) if isinstance(batch["target"], Tensor) else torch.tensor(batch["target"], device=device)
        snrs = batch["snr_db"].cpu().numpy() if isinstance(batch["snr_db"], Tensor) else np.array(batch["snr_db"])

        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(wavs).logits

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_snrs.extend(snrs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_snrs = np.array(all_snrs)

    overall_acc = float((all_preds == all_targets).mean())

    snr_results = {}
    for snr in SNR_LEVELS:
        mask = np.isclose(all_snrs, snr, atol=0.5)
        if mask.sum() > 0:
            snr_acc = float((all_preds[mask] == all_targets[mask]).mean())
            snr_results[f"SNR_{int(snr):+d}dB"] = snr_acc

    return {
        "overall_acc": overall_acc,
        "snr_breakdown": snr_results,
        "total_samples": len(all_preds)
    }
'''

# 4. New KVrH (train loop using .logits)
new_kvrh = r'''# Vòng Lặp Huấn Luyện BEAT-BEAT KD Logit
def run_experiment(is_full_train: bool, batch_size: int, lr: float, temp: float, kd_w: float, epochs: int):
    max_s = None if is_full_train else 128
    num_ep = epochs if is_full_train else 1

    train_ds = Mixture36Dataset(DATASET_DIR, split="train", max_samples=max_s)
    val_ds = Mixture36Dataset(DATASET_DIR, split="eval", max_samples=max_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = BornAgainLogitLoss(temperature=temp, kd_weight=kd_w, label_smoothing=0.1)
    optimizer = AdamW(student_model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))

    history = []
    best_acc = 0.0
    out_dir = Path("/marimo/output/beats-kd-logit")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Bắt đầu chạy: Mode={'FULL' if is_full_train else 'SMOKE'} | Epochs={num_ep} | Batch={batch_size} | LR={lr} | T={temp} | λ={kd_w}\n")

    for epoch in range(1, num_ep + 1):
        student_model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0

        for b_idx, batch in enumerate(train_loader):
            wavs = batch["waveform"].to(device)
            targets = batch["target"].to(device) if isinstance(batch["target"], Tensor) else torch.tensor(batch["target"], device=device)

            # 1. Forward Teacher (eval mode, no grad)
            with torch.no_grad():
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    t_logits = teacher_model(wavs).logits

            # 2. Forward Student
            with torch.autocast(device.type, dtype=torch.bfloat16):
                s_logits = student_model(wavs).logits
                loss, terms = criterion(s_logits, t_logits, targets)

            # 3. Backward & Update Student
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += loss.item() * len(targets)
            correct += (s_logits.argmax(-1) == targets).sum().item()
            total += len(targets)

        # Đánh giá sau epoch
        val_res = evaluate_model(student_model, val_loader, device)
        train_acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / total if total > 0 else 0.0
        ep_time = time.time() - t0

        if val_res["overall_acc"] > best_acc:
            best_acc = val_res["overall_acc"]
            torch.save({
                "epoch": epoch,
                "student_state": student_model.state_dict(),
                "best_acc": best_acc
            }, out_dir / "best_student.pt")

        ep_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_res["overall_acc"], 4),
            "time_s": round(ep_time, 1),
            **{k: round(v, 4) for k, v in val_res["snr_breakdown"].items()}
        }
        history.append(ep_log)
        print(f"Epoch {epoch}/{num_ep} ({ep_time:.1f}s): {ep_log}")

    return pd.DataFrame(history)

btn_train = mo.ui.run_button(label="Bắt đầu chạy huấn luyện (Click to Execute)")
btn_train
'''

# 5. New xpMA (display results)
new_xpma = r'''# Hiển thị Kết quả
res_df = None
if btn_train.value:
    res_df = run_experiment(
        is_full_train=ui_run_training.value,
        batch_size=int(ui_batch_size.value),
        lr=float(ui_lr.value),
        temp=float(ui_temperature.value),
        kd_w=float(ui_kd_weight.value),
        epochs=int(ui_epochs.value)
    )

if res_df is not None:
    mo.vstack([
        mo.md("### 📊 Kết Quả Huấn Luyện & Đánh Giá theo SNR"),
        mo.ui.table(res_df)
    ])
else:
    mo.md("*Nhấn nút **'Bắt đầu chạy huấn luyện'** ở trên để kích hoạt vòng chạy.*")
'''

edits = [
    ("IBDh", new_ibdh, False),
    ("HcAs", new_hcas, False),
    ("vOZk", new_vozk, False),
    ("KVrH", new_kvrh, False),
    ("xpMA", new_xpma, False),
]

script_lines = [
    "import base64",
    "import marimo._code_mode as cm",
    "",
    "async with cm.get_context() as ctx:",
]

for cid, code, hide in edits:
    b64_val = base64.b64encode(code.encode("utf-8")).decode("ascii")
    script_lines.append(f"    ctx.edit_cell({cid!r}, code=base64.b64decode({b64_val!r}).decode('utf-8'), hide_code={hide})")
    script_lines.append(f"    ctx.run_cell({cid!r})")

script_lines.append("    print('ALL_CELLS_FIXED_SUCCESSFULLY')")

out_text = "\n".join(script_lines)
Path("d:/PROJECT/MLNC/scratch/fix_cells_runner.py").write_text(out_text, encoding="ascii")
print(f"Generated fix script: {len(out_text)} chars")
