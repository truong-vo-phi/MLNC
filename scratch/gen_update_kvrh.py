import base64
from pathlib import Path

new_kvrh = r'''# Vòng Lặp Huấn Luyện BEAT-BEAT KD Logit
def run_experiment(is_full_train: bool, batch_size: int, lr: float, temp: float, kd_w: float, epochs: int):
    max_s = None if is_full_train else 128
    num_ep = epochs if is_full_train else 1

    train_ds = Mixture36Dataset(DATASET_DIR, split="train", max_samples=max_s)
    val_ds = Mixture36Dataset(DATASET_DIR, split="eval", max_samples=max_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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

b64_val = base64.b64encode(new_kvrh.encode("utf-8")).decode("ascii")
script = f"""import base64
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    ctx.edit_cell('KVrH', code=base64.b64decode({b64_val!r}).decode('utf-8'), hide_code=False)
    ctx.run_cell('KVrH')
    ctx.run_cell('xpMA')
    print('KVrH updated to num_workers=0 successfully!')
"""
Path("d:/PROJECT/MLNC/scratch/update_kvrh.py").write_text(script, encoding="ascii")
