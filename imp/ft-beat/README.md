# FT-BEAT: Fine-tuning BEATs trên Hỗn hợp Âm thanh qua Bộ Tách Nhiễu FRCRN

Gói mã nguồn tự chứa tại `imp/ft-beat`, triển khai phương pháp **Fine-tuning mô hình BEATs (ViT-Base 90.34M tham số) trên tín hiệu nhiễu ước lượng $\hat{\mathbf{n}}$ được tách ra từ hỗn hợp âm thanh (Mixture) thông qua mô hình FRCRN đã huấn luyện trước và đóng băng**.

Hệ thống được thiết kế theo chuẩn Baseline Đợt 4 (`run1_baseline`), khớp với các cấu hình và kết quả thực nghiệm chuẩn trong bảng so sánh 6 mức SNR (-5 dB đến 20 dB) của dự án MLNC.

---

## 1. Cơ sở Lý thuyết & Sơ đồ Luồng Xử lý

### 1.1. Luồng xử lý tín hiệu (Inference & Training Pipeline)

```
Mixture Audio x_mix (16kHz, 4.0s)
          │
          ▼
┌─────────────────────────────────┐
│ FRCRN Separator (Đóng băng)     │ ──► Ước lượng tiếng nói ŝ
│ Complex UNet + STFT Masking     │ ──► Nhiễu n̂ = x_mix - ŝ
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Chuẩn hóa RMS (target_rms=0.08) │ ──► n̂_norm = n̂ * (target_rms / (RMS(n̂) + ε))
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Log-Mel Filterbank (128 bins)   │ ──► Mel Spectrogram [B, 128, 398]
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ BEATs ViT-Base Encoder (12 blk) │ ──► Patch Embed 16x16 + 12 Transformer Layers
│ + Linear Head (36 lớp nhiễu)    │ ──► Logits [B, 36]
└─────────────────────────────────┘
          │
          ▼
   Cross-Entropy Loss
```

### 1.2. Công thức Toán học

1. **Tách tín hiệu nhiễu ước lượng:**
   $$\hat{\mathbf{s}} = \text{FRCRN}(\mathbf{x}_{\text{mix}}), \quad \hat{\mathbf{n}} = \mathbf{x}_{\text{mix}} - \hat{\mathbf{s}}$$

2. **Chuẩn hóa RMS (Energy Normalization):**
   Đưa độ lớn năng lượng của clip nhiễu về phân phối chuẩn của AudioSet ($\sigma_0 \approx 0.07925$):
   $$\text{RMS}(\hat{\mathbf{n}}) = \sqrt{\frac{1}{T} \sum_{t=1}^T \hat{n}_t^2}$$
   $$\hat{\mathbf{n}}_{\text{norm}} = \hat{\mathbf{n}} \cdot \frac{\sigma_0}{\text{RMS}(\hat{\mathbf{n}}) + \epsilon}$$

3. **Huấn luyện phân loại với Cross-Entropy Loss:**
   $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} y_c \log \frac{\exp(z_c)}{\sum_{j=1}^C \exp(z_j)}$$
   với $C = 36$ lớp nhiễu môi trường, tối ưu hóa bằng AdamW kết hợp lịch học Cosine Warmup.

---

## 2. Cấu trúc Thư mục

```
imp/ft-beat/
├── config/
│   └── ft_beat.yaml        # Cấu hình siêu tham số huấn luyện & đường dẫn dữ liệu
├── ft_beat/
│   ├── __init__.py         # Export API chính
│   ├── config.py           # Dataclass FTBeatConfig kế thừa BaseConfig
│   ├── data.py             # Nạp Mixture-36l, cân bằng nhãn, cache n̂, lọc Unseen
│   ├── evaluate.py         # Đo đạc 9 chỉ số tổng, chi tiết theo SNR, bộ lọc Unseen
│   ├── model.py            # FRCRNBEATPipeline (FRCRN + BEATs + RMS normalize)
│   ├── pipeline.py         # Quy trình huấn luyện, lưu vết, kiểm định và trích xuất
│   ├── runtime.py          # Quản lý thiết bị và mixed-precision autocast (bf16/fp16)
│   └── trainer.py          # FTBeatTrainer kế thừa BEATTrainer (hỗ trợ grad accumulation)
├── tests/
│   ├── conftest.py         # Cấu hình đường dẫn pytest
│   ├── test_config.py      # Kiểm thử tính hợp lệ của cấu hình
│   ├── test_data.py        # Kiểm thử sampler cân bằng và lọc Unseen
│   ├── test_model.py       # Kiểm thử chuẩn hóa RMS và forward pipeline
│   └── test_trainer.py     # Kiểm thử chu kỳ fit và khôi phục resume bit-for-bit
├── run.py                  # Điểm vào dòng lệnh (CLI)
└── README.md               # Tài liệu này
```

---

## 3. Cấu hình Siêu Tham số Huấn luyện

Dựa trên cấu hình chuẩn của **Đợt 4 – BEATs Baseline (`run1_baseline`)**:

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| **Backbone** | BEATs ViT-Base (iter3+ AS2M) | 90.34M tham số |
| **Frontend** | FRCRN (epoch 37 / best) | 13.14M tham số (đóng băng) |
| **Số lớp phân loại** | 36 lớp | Phân loại đơn nhãn / multi-hot |
| **Tần số lấy mẫu** | 16,000 Hz | Độ dài clip: 4.0 giây |
| **Dải SNR huấn luyện** | [-5, 0, 5, 10, 15, 20] dB | 6 mức SNR trên `mixture-36l` |
| **Batch size hiệu dụng** | 32 | Batch size 16 × Tích lũy Gradient 2 bước |
| **Tốc độ học (Learning Rate)** | 1.0e-4 | Giảm dần theo Cosine Annealing, warmup 10% |
| **Trọng số suy giảm (Weight Decay)**| 0.01 | AdamW ($\beta_1=0.9, \beta_2=0.98$) |
| **Layer Decay** | 0.3 | Giảm LR theo tầng sâu của ViT |
| **Số epoch tối đa** | 10 epochs | Dừng sớm với patience = 4 |
| **Tiêu chí chọn Checkpoint** | `macro_f1` | Đánh giá trên toàn bộ tập validation |

---

## 4. Hướng dẫn Sử dụng

### 4.1. Kiểm tra tham số mô hình & bộ nhớ
```powershell
python run.py params
```

### 4.2. Chạy Smoke Test (kiểm tra toàn bộ luồng huấn luyện trên CPU/GPU)
```powershell
python run.py smoke
```

### 4.3. Huấn luyện chính thức
```powershell
python run.py train
```
Tiếp tục huấn luyện từ phiên dở dang (Bit-for-Bit Resume):
```powershell
python run.py train --run-id ft-beat-2026-10-29-16-00-00
```

### 4.4. Đánh giá Checkpoint có sẵn
```powershell
python run.py evaluate --checkpoint output/checkpoint/ft-beat-xxx/beats/last_best_checkpoint.pt
```

### 4.5. Xuất mô hình phục vụ Inference
```powershell
python run.py export --checkpoint output/checkpoint/ft-beat-xxx/beats/last_best_checkpoint.pt --output output/beats_ft_deployed.pt
```

---

## 5. Kiểm thử Tự động (Unit Tests)

Chạy bộ kiểm thử tự động đảm bảo chất lượng:
```powershell
python -m pytest tests
```
Tất cả 11 bài kiểm thử bao phủ toàn bộ các module: cấu hình, chuẩn hóa dữ liệu, pipeline mô hình và chu trình tối ưu hóa.

---

## 6. Bảng Kết quả Tham chiếu (Benchmark Reference)

Kết quả tham chiếu từ Đợt 4 (BEATs Baseline `run1_baseline`):

### 6.1. Toàn bộ Validation (All SNR - 6,480 mẫu)
- **Top-1 Accuracy:** `63.66%`
- **Top-3 Accuracy:** `85.65%`
- **Balanced Accuracy:** `63.66%`
- **Precision Macro:** `65.37%`
- **Recall Macro:** `63.66%`
- **Macro-F1:** `63.48%`
- **mAP:** `69.52%`
- **Macro-AUC:** `95.81%`

### 6.2. Validation Unseen (Lọc bỏ YTID AudioSet Train - 3,064 mẫu)
- **Top-1 Accuracy:** `58.88%`
- **Top-3 Accuracy:** `75.23%`
- **Balanced Accuracy:** `56.30%`
- **Precision Macro:** `57.76%`
- **Recall Macro:** `56.30%`
- **Macro-F1:** `55.50%`
- **mAP:** `61.60%`
- **Macro-AUC:** `94.01%`
