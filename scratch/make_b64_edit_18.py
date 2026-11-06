import base64

cell18_code = """from pathlib import Path as _Path
import json as _json

MOE_LOW_CHECKPOINT_AFTER_TRAIN = MOE_OUTPUT_ROOT / "low/last_best_checkpoint.pt"
MOE_HIGH_CHECKPOINT_AFTER_TRAIN = MOE_OUTPUT_ROOT / "high/last_best_checkpoint.pt"
MOE_INFERENCE_READY_AFTER_TRAIN = MOE_TRAINING_COMPLETE and MOE_LOW_CHECKPOINT_AFTER_TRAIN.is_file() and MOE_HIGH_CHECKPOINT_AFTER_TRAIN.is_file()

print("INFERENCE_READY_AFTER_TRAIN=" + str(MOE_INFERENCE_READY_AFTER_TRAIN))
print("Low checkpoint:", MOE_LOW_CHECKPOINT_AFTER_TRAIN)
print("High checkpoint:", MOE_HIGH_CHECKPOINT_AFTER_TRAIN)

if MOE_INFERENCE_READY_AFTER_TRAIN:
    MOE_STAGE_SUMMARIES = {
        "Low Student": _json.loads((MOE_OUTPUT_ROOT / "low/summary.json").read_text(encoding="utf-8")),
        "High Student": _json.loads((MOE_OUTPUT_ROOT / "high/summary.json").read_text(encoding="utf-8")),
    }
    def _best_validation_metrics(_summary):
        _epoch = int(_summary.get("best_epoch", -1))
        _row = next((r for r in _summary.get("history", []) if int(r.get("epoch", -2)) == _epoch), None)
        if _row is None:
            return {}
        return {k[4:]: float(v) for k, v in _row.items() if k.startswith("val_") and isinstance(v, (int, float))}
    
    MOE_STAGE_METRICS = {stage: _best_validation_metrics(summary) for stage, summary in MOE_STAGE_SUMMARIES.items()}
    
    _primary_metrics = [
        ("Top-1 Accuracy (ACC)", "acc", "63.58%"),
        ("Top-3 Accuracy", "top3_accuracy", "80.19%"),
        ("Balanced Accuracy", "balanced_accuracy", "62.03%"),
        ("Macro Precision", "precision_macro", "—"),
        ("Macro Recall", "recall_macro", "—"),
        ("Macro-F1", "macro_f1", "—"),
        ("Micro-F1", "micro_f1", "—"),
        ("mAP (Overall)", "mAP", "69.57%"),
        ("Macro AUC", "macro_auc", "96.19%"),
        ("Mean Balanced Acc across SNRs", "macro_acc", "—"),
        ("Mean mAP across SNRs", "mAP_macro", "70.08%"),
    ]
    def _fmt_metric(_metrics, _key):
        _val = _metrics.get(_key)
        return "N/A" if _val is None else f"{100.0 * _val:.2f}%"

    _lines = [
        "## 🏆 Kết Quả Đánh Giá Huấn Luyện MoE Students", "",
        "> 📊 **Ghi chú:** Kết quả trích xuất từ epoch kiểm định tốt nhất của từng Student (`Low-Student` theo `macro_acc`, `High-Student` theo `top1_accuracy`).", "",
        f"| Chỉ số | Low Student (Epoch {MOE_STAGE_SUMMARIES['Low Student'].get('best_epoch')}) | High Student (Epoch {MOE_STAGE_SUMMARIES['High Student'].get('best_epoch')}) | Mục tiêu Champion |",
        "|---|---:|---:|---:|",
    ]
    for _label, _key, _tgt in _primary_metrics:
        _lines.append(f"| **{_label}** | {_fmt_metric(MOE_STAGE_METRICS['Low Student'], _key)} | {_fmt_metric(MOE_STAGE_METRICS['High Student'], _key)} | `{_tgt}` |")

    _lines.extend([
        "",
        "### 📈 Phân Tích Chi Tiết Theo Từng Mức Nhiễu (SNR Breakdown)", "",
        "| SNR Level | Low Acc | Low Bal. Acc | Low Macro-F1 | Low mAP | High Acc | High Bal. Acc | High Macro-F1 | High mAP |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _snr in (-5, 0, 5, 10, 15, 20):
        _low = MOE_STAGE_METRICS["Low Student"]
        _high = MOE_STAGE_METRICS["High Student"]
        _keys = (f"top1_accuracy_snr_{_snr}", f"balanced_accuracy_snr_{_snr}", f"macro_f1_snr_{_snr}", f"mAP_snr_{_snr}")
        _low_cols = [_fmt_metric(_low, k) for k in _keys]
        _high_cols = [_fmt_metric(_high, k) for k in _keys]
        _lines.append(f"| **{_snr} dB** | " + " | ".join(_low_cols + _high_cols) + " |")

    _lines.extend([
        "",
        "> 💡 **Đánh giá tuyến tính:** Low-Student tối ưu hóa cho vùng nhiễu mạnh ($\\le 15\\text{ dB}$), trong khi High-Student chuyên biệt hóa xuất sắc tại SNR cao ($20\\text{ dB}$).",
        f"> **Checkpoint đã lưu:**",
        f"> - Low: `{MOE_LOW_CHECKPOINT_AFTER_TRAIN}`",
        f"> - High: `{MOE_HIGH_CHECKPOINT_AFTER_TRAIN}`"
    ])
    MOE_EVALUATION_VIEW = mo.md(chr(10).join(_lines))
else:
    MOE_EVALUATION_VIEW = mo.md(\"\"\"
### ⏳ Chờ Huấn Luyện Hoàn Tất
*Bảng kết quả đánh giá chi tiết (Top-1, mAP, Balanced Acc, và phân tách 6 mốc SNR) sẽ tự động hiển thị sau khi cả 2 sinh viên Low và High hoàn tất quá trình huấn luyện.*
\"\"\")

MOE_EVALUATION_VIEW
"""

b64_18 = base64.b64encode(cell18_code.encode("utf-8")).decode("ascii")

remote_script = f"""import base64
import marimo._code_mode as cm

code18 = base64.b64decode("{b64_18}").decode("utf-8")
async with cm.get_context() as ctx:
    ctx.edit_cell("TqIu", code=code18)
    ctx.run_cell("TqIu")

print("SUCCESS_EDIT_TQIU")
"""

with open("d:/PROJECT/MLNC/scratch/run_remote_edit_18.py", "w", encoding="ascii") as f:
    f.write(remote_script)
print("Wrote run_remote_edit_18.py")
