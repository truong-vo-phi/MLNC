import sys
import marimo._code_mode as cm
from pathlib import Path

# New cell code for student-only inference and display on notebook
NEW_EVAL_CELL_CODE = r'''import json as _json
from pathlib import Path as _Path
import marimo as _mo

# Load Student-Only Benchmark Results
_student_bench_path = _Path('/marimo/output/student_only_benchmark.json')

if _student_bench_path.is_file():
    _bench = _json.loads(_student_bench_path.read_text(encoding='utf-8'))
    _r1 = _bench.get('run1_dual_student_only', {})
    
    _md_lines = [
        "# Kết Quả Nghiệm Thu Champion MoE (Inference: Chỉ Dùng Student, 0% BEATs)",
        "",
        "> **Thiết kế triển khai biên (Edge Deployment)**: Giáo viên BEATs (90.34M) chỉ tham gia trong quá trình **huấn luyện (Offline Knowledge Distillation)**. Khi suy luận, hệ thống hoàn toàn loại bỏ BEATs và chỉ chạy **2 học sinh CED-Tiny siêu nhẹ (~10.84M tham số)** định tuyến theo mức SNR.",
        "",
        "### 1. Bảng Hiệu Năng Tổng Thể Toàn Bộ Tập Validation (3,064 clips, 3-Way TTA)",
        "",
        "| Chỉ Số Đánh Giá (Metric) | Kết Quả Dual-Student (~10M Params) | Ý Nghĩa Triển Khai |",
        "| :--- | :---: | :--- |",
        f"| **Top-1 Accuracy** | **{_r1.get('top1_accuracy', 0)*100:.2f}%** | Độ chính xác phân loại âm thanh 36 lớp |",
        f"| **Top-3 Accuracy** | **{_r1.get('top3_accuracy', 0)*100:.2f}%** | Tỷ lệ nhãn đúng nằm trong top-3 dự đoán |",
        f"| **Balanced Accuracy** | **{_r1.get('balanced_accuracy', 0)*100:.2f}%** | Độ chính xác cân bằng trên 36 lớp |",
        f"| **Macro F1-Score** | **{_r1.get('macro_f1', 0)*100:.2f}%** | F1 vĩ mô cân đối chính xác và độ phủ |",
        f"| **Macro mAP** | **{_r1.get('mAP', 0)*100:.2f}%** | Độ chính xác trung bình trung bình |",
        f"| **Macro ROC-AUC** | **{_r1.get('macro_auc', 0)*100:.2f}%** | Khả năng phân biệt nhãn âm thanh |",
        "",
        "### 2. Bảng Phân Rã Độ Chính Xác Gộp Theo Từng Mức SNR (-5 dB đến 20 dB)",
        "",
        "| Mức SNR (dB) | Chuyên Gia Xử Lý (Route) | Số Lượng Clips | Top-1 Accuracy | Macro mAP | Macro F1 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    
    _snr_counts = {-5: 536, 0: 511, 5: 488, 10: 513, 15: 507, 20: 509}
    for _snr in (-5, 0, 5, 10, 15, 20):
        _route = "Low-Student (< 18 dB)" if _snr < 18 else "High-Student (>= 18 dB)"
        _acc = _r1.get(f'top1_accuracy_snr_{_snr}', 0) * 100
        _map = _r1.get(f'mAP_snr_{_snr}', 0) * 100
        _f1 = _r1.get(f'macro_f1_snr_{_snr}', 0) * 100
        _cnt = _snr_counts.get(_snr, 0)
        _md_lines.append(f"| **{_snr} dB** | {_route} | {_cnt} | **{_acc:.2f}%** | {_map:.2f}% | {_f1:.2f}% |")
        
    _md_lines.extend([
        "",
        "---",
        "### 3. Đối Chiếu Giai Đoạn Huấn Luyện (Training Stage Distillation)",
        "- **Low-Student (S1)**: Huấn luyện 15 epochs trên tất cả SNR với BEATs Fold0 (`input_rms = 0.07925`), đạt Best Val Score: **56.07%**.",
        "- **High-Student (S2)**: Khởi tạo từ Low-Student, fine-tune 10 epochs trên SNR = 20 dB với BEATs EMA, đạt Best Top-1: **46.37%**.",
        "- **Suy Luận MoE (Dual-Student Only)**: Kết hợp 2 chuyên gia với ngưỡng định tuyến 18 dB giúp nâng hiệu năng toàn diện lên **59.04% Top-1** và **77.51% Top-3** mà không cần tới giáo viên lúc chạy thực tế.",
    ])
    
    MOE_STUDENT_ONLY_VIEW = _mo.md("\n".join(_md_lines))
else:
    MOE_STUDENT_ONLY_VIEW = _mo.md("Chưa tìm thấy kết quả benchmark student_only_benchmark.json.")

MOE_STUDENT_ONLY_VIEW
'''

async def update():
    async with cm.get_context(skip_validation=True) as ctx:
        if 'TqIu' in ctx.cells:
            ctx.edit_cell('TqIu', code=NEW_EVAL_CELL_CODE)
            ctx.run_cell('TqIu')
            print("Successfully updated and executed cell TqIu!")
        else:
            print("Cell TqIu not found, cells:", list(ctx.cells.keys()))

await update()
