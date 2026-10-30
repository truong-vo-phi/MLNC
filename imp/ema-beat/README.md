# ema-beat: Fine-tuning BEATs với Exponential Moving Average (EMA)

Dự án tự chứa trong `imp/ema-beat`, hiện thực hóa **Ý tưởng 1: Model EMA (Polyak/Weights Averaging) cho quá trình Fine-tuning mô hình BEATs**. Pipeline hỗ trợ fine-tune backbone BEATs (ViT-Base 90M tham số) trên tập dữ liệu phân loại âm thanh (`mixture-36l`), kết hợp theo dõi mô hình shadow EMA để giảm phương sai tối ưu, chống overfitting và nâng cao độ chính xác mà **hoàn toàn không làm tăng chi phí suy luận (Zero Inference Overhead)**.

---

## 1. Cơ Sở Lý Thuyết & Công Thức Toán Học

### 1.1. Công thức cập nhật trọng số EMA
Mô hình chính (Student/Active Model $\theta_{\text{train}}$) được tối ưu bằng AdamW qua backpropagation. Song song đó, một mô hình bóng (**Shadow Model $\theta_{\text{EMA}}$**) được cập nhật sau mỗi optimizer step:

$$\theta_{\text{EMA}}^{(t)} = \beta(t) \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \beta(t)) \cdot \theta_{\text{train}}^{(t)}$$

Trong đó:
- $\beta(t) \in [0.90, 0.9999]$ là hệ số suy giảm (decay factor).
- Với $\beta \approx 0.9995$, trọng số EMA đóng vai trò như trung bình trọng số của khoảng $\frac{1}{1 - \beta} \approx 2000$ bước tối ưu gần nhất, làm phẳng các hố cực tiểu nhọn (sharp minima) trên hàm mất mát và hội tụ về vùng cực tiểu phẳng (flat minima).

### 1.2. Decay Warmup Schedule
Ở các bước đầu tiên, trọng số $\theta_{\text{train}}$ biến đổi rất nhanh do learning rate cao và gradient lớn. Nếu áp dụng ngay decay quá cao (0.9999), mô hình EMA sẽ bị neo lại ở trạng thái khởi tạo ngẫu nhiên quá lâu. Do đó, hệ số $\beta$ được tăng dần theo lịch trình:

$$\beta(t) = \beta_0 + (\beta_{\text{target}} - \beta_0) \cdot \min\left(1, \frac{t}{\tau_{\text{warmup}}}\right)$$

- $\beta_0 = 0.90$: Cho phép EMA cập nhật nhanh ở các batch đầu tiên.
- $\beta_{\text{target}} = 0.9995$: Đảm bảo độ mượt tối đa khi mô hình đã dần ổn định.
- $\tau_{\text{warmup}} = 1000$ steps.

### 1.3. Khoảng cách trọng số L2
Để theo dõi mức độ phân kỳ hoặc ổn định của mô hình đang huấn luyện so với shadow EMA:

$$D_{\text{EMA}}(t) = \|\theta_{\text{train}}^{(t)} - \theta_{\text{EMA}}^{(t)}\|_2 = \sqrt{\sum_{i} \|\theta_{\text{train}, i}^{(t)} - \theta_{\text{EMA}, i}^{(t)}\|^2_2}$$

---

## 2. Cấu Trúc Thư Mục

```
imp/ema-beat/
├── ema_beat/
│   ├── __init__.py           # Export các thành phần chính
│   ├── ema.py                # Lớp BEATsModelEMA: quản lý shadow weights, warmup, CPU offload
│   ├── config.py             # Dataclass BaseConfig: EMAConfig, EMABeatTrainConfig, v.v.
│   ├── trainer.py            # EMABeatTrainer kế thừa BEATTrainer: dual evaluation & checkpointing
│   ├── evaluate.py           # Đánh giá song song Raw vs EMA, phân rã theo SNR, lập bảng so sánh
│   ├── pipeline.py           # Luồng thực thi hoàn chỉnh từ nạp dữ liệu đến huấn luyện và xuất file
│   ├── cli.py                # Điểm vào dòng lệnh (train, smoke, params, export)
│   └── runtime.py            # Tiện ích device, autocast (bf16/fp16/fp32)
├── config/
│   └── ema_beat.yaml         # File cấu hình yaml chuẩn của thí nghiệm
├── tests/
│   ├── conftest.py           # Thiết lập đường dẫn import cho pytest
│   ├── test_config.py        # Kiểm thử tính hợp lệ của cấu hình
│   ├── test_ema.py           # Kiểm thử độ chính xác số học của cập nhật EMA
│   └── test_trainer.py       # Kiểm thử huấn luyện 1-2 epoch và tính năng resume
├── run.py                    # Script chạy chính
├── eat-beat.ipynb            # Notebook tương tác trực quan minh họa kiến trúc & chạy thử
└── README.md                 # Tài liệu này
```

---

## 3. Các Tính Năng Nổi Bật Của `EMABeatTrainer`

1. **Dual Evaluation (Đánh giá song song):**
   Mỗi epoch, trainer đánh giá cả mô hình gốc (`val_raw_acc`, `val_raw_macro_f1`) và mô hình EMA (`val_ema_acc`, `val_ema_macro_f1`) trên cùng một tập validation, giúp quan sát trực tiếp mức độ cải thiện của EMA theo thời gian thực.
2. **Lựa chọn Checkpoint linh hoạt:**
   Tham số `selection_target: ema` (mặc định) đảm bảo mô hình tốt nhất được chọn và xuất theo chỉ số cao nhất của EMA.
3. **Resume bit-for-bit:**
   File `resume.pt` lưu trữ đầy đủ `step_count`, trọng số của cả Student và Shadow EMA, optimizer và learning rate scheduler.
4. **Standalone Model Export:**
   Cung cấp tính năng bóc tách trọng số EMA thành một checkpoint `BEATClassifier` chuẩn, có thể đem đi deploy ở bất kỳ đâu mà không cần phụ thuộc vào mã nguồn EMA.

---

## 4. Hướng Dẫn Sử Dụng

### 4.1. Kiểm tra tham số và bộ nhớ
```powershell
python run.py params
```

### 4.2. Chạy Smoke Test kiểm tra pipeline (nhanh trên CPU/GPU)
```powershell
python run.py smoke
```

### 4.3. Huấn luyện thật trên bộ dữ liệu Mixture-36l
```powershell
python run.py train
```
Tiếp tục một phiên huấn luyện dở dang:
```powershell
python run.py train --run-id ema-beat-2026-10-29-10-00-00
```

### 4.4. Bóc tách mô hình EMA phục vụ Inference
```powershell
python run.py export --checkpoint output/checkpoint/ema-beat-xxx/beats_ema/resume.pt --output output/beats_ema_deployed.pt
```

---

## 5. Chạy Kiểm Thử (Unit Tests)
```powershell
python -m pytest tests
```
Tất cả 13 bài kiểm thử unit tests đều bao phủ kiểm tra số học, cấu hình, và các hook trong trainer.
