# Champion Pipeline: Dual-Expert & Dual-Teacher Mixture of Experts (MoE)

Hệ thống phân loại âm thanh kháng nhiễu cực đại (36 lớp) kết hợp **Dual-Expert CED-Tiny** (BEATs 2D Relative Position Bias), **Dual-Teacher BEATs 12L**, **3-Way Temporal TTA**, **Hiệu chuẩn nhiệt độ không đối xứng** và **Định tuyến động theo mức năng lượng SNR**.

---

## 1. Kết Quả Benchmark Trên Tập Kiểm Định Sạch (3,064 Clips)

| Chỉ số Toàn cục | Kết quả Đạt Được | Mục tiêu Đề tài | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **`63.58%`** | $0.62 - 0.63$ | **ĐẠT & VƯỢT** |
| **Top-3 Accuracy** | **`80.19%`** | $> 75\%$ | **ĐỘT PHÁ** |
| **Balanced Accuracy** | **`62.03%`** | — | **TỐI ƯU** |
| **Macro mAP** | **`70.08%`** | $\ge 0.70$ | **CHÍNH THỨC ĐẠT** |
| **Overall mAP** | **`69.57%`** | — | **TỐI ƯU** |
| **Macro AUC** | **`96.19%`** | $> 92\%$ | **KỶ LỤC** |

### Chi tiết hiệu năng theo từng mốc SNR

| Mốc SNR | Số mẫu | Top-1 Accuracy | Macro mAP | Ghi chú chuyên môn |
| :---: | :---: | :---: | :---: | :--- |
| **$-5\text{ dB}$** | 500 clips | **72.20%** | **78.09%** | Tuyến Low: Fold0 (45%) + Student Low (30%) đạt đỉnh |
| **$0\text{ dB}$** | 518 clips | **71.62%** | **77.41%** | Tuyến Low: Duy trì ổn định trên 71% |
| **$5\text{ dB}$** | 505 clips | **64.55%** | **73.52%** | Tuyến Low: Phân tách 36 lớp vượt trội |
| **$10\text{ dB}$** | 513 clips | **62.96%** | **67.71%** | Tuyến Low: Giữ vững độ chính xác cao |
| **$15\text{ dB}$** | 519 clips | **57.79%** | **67.05%** | Tuyến Low: Vùng chuyển tiếp |
| **$20\text{ dB}$** | 509 clips | **51.87%** | **56.69%** | Tuyến High: EMA (40%) + Student High (35%) bứt phá +6.88% |

---

## 2. Kiến Trúc Hệ Thống (End-to-End Pipeline)

```
                            x_mix [B, 64000]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              FRCRN (13.14M)                SNR Router (τ = 18 dB)
                    │                             │
            n̂ = x_mix - ŝ                         │
                    │                             │
         ┌──────────┴──────────┐                  │
         ▼          ▼          ▼                  │
        w_0       w_+50      w_-50                │
   (3-Way Temporal TTA, RMS=0.07925)              │
         │          │          │                  │
         └──────────┬──────────┘                  │
                    ▼                             │
    ┌───────────────────────────────┐             │
    │   4 Parallel Neural Engines   │             │
    ├───────────────────────────────┤             │
    │ S1: Low-Student (5.42M)       │◄───(SNR < 18 dB)
    │ S2: High-Student (5.42M)      │◄───(SNR ≥ 18 dB)
    │ T1: Teacher Fold0 (90.34M)    │             │
    │ T2: Teacher EMA (90.34M)      │             │
    └───────────────┬───────────────┘             │
                    ▼                             │
       TTA Logit Aggregation                      │
   z = (z_0 + 0.5*z_+50 + 0.5*z_-50) / 2          │
                    │                             │
                    ▼                             │
        Temperature Calibration                   │
   p_s = Softmax(z_s / 1.40)                      │
   p_t = Softmax(z_t / 1.20)                      │
                    │                             │
                    ▼                             ▼
         Dynamic Routing Blending ◄───────────────┘
     ┌─────────────────────────────────────────┐
     │ Route 1 (SNR < 18 dB):                  │
     │   P = 0.30·p_s_low + 0.45·p_t0 + 0.25·p_tema
     ├─────────────────────────────────────────┤
     │ Route 2 (SNR ≥ 18 dB):                  │
     │   P = 0.35·p_s_high + 0.40·p_tema + 0.25·p_t0
     └────────────────────┬────────────────────┘
                          ▼
            Final Output: ŷ = Argmax(P)
```

---

## 3. Thống Kê Tham Số Hệ Thống

| Khối / Module | Cấu hình | Tham số |
| :--- | :--- | :---: |
| **FRCRN Frontend** | Complex UNet + FSMN | `13.14M` (Pre-cached) |
| **CED-Tiny Low-Expert (S1)** | 12L, D=192, H=3, 2D Relative Bias | `5.42M` |
| **CED-Tiny High-Expert (S2)** | 12L, D=192, H=3, Fine-tuned 20 dB | `5.42M` |
| **Teacher BEATs 12L Fold0 (T1)** | 12L, D=768, H=12, Clean Noise FT | `90.34M` |
| **Teacher BEATs 12L EMA (T2)** | 12L, D=768, H=12, FRCRN Noise EMA | `90.34M` |
| **Tổng số tham số 4 động cơ** | S1 + S2 + T1 + T2 | **`191.53M`** |
| **Nhánh Sinh viên Độc lập** | S1 + S2 | **`10.84M`** |

---

## 4. Cấu Trúc Thư Mục `imp/moe/`

```
imp/moe/
├── config/
│   ├── moe.yaml             # Cấu hình chính của Champion Pipeline
│   ├── beat.yaml            # Cấu hình kiến trúc Teacher BEATs 12L
│   └── url.yaml             # Cấu hình URL và đường dẫn pretrain
├── src/
│   ├── pipeline.py          # Class ChampionPipeline, load_pipeline, evaluate_pipeline
│   ├── router.py            # EnergySNRRouter & NeuralRouter (Soft Gating)
│   ├── metrics.py           # Tính toán 9 metrics và snr_scores breakdown
│   ├── model/
│   │   ├── student/         # CEDStudent, vit_bias (2D Gated Bias), frontend LogMel
│   │   └── beat/            # BEATClassifier, checkpoint loader, attention
│   ├── core/                # Module base & config base
│   └── utils/               # RMS normalize, seed, paths
├── scripts/
│   └── load_dataset.py      # Mixture36Dataset, NoiseCacheDataset, build_dataloader
├── tests/
│   ├── test_pipeline.py     # Unit test cho pipeline và routing
│   ├── test_router.py       # Unit test cho router năng lượng và neural router
│   └── test_student.py      # Unit test cho CEDStudent 2D relative bias
├── champion_pipeline.py     # CLI chính thực thi và đánh giá Pipeline
├── run.py                   # Shortcut runner
└── README.md                # Tài liệu kỹ thuật
```

---

## 5. Hướng Dẫn Sử Dụng

### Đánh giá Pipeline trên GPU / Sandbox:
```bash
python champion_pipeline.py \
  --config config/moe.yaml \
  --low-ckpt /marimo/output/kd-v5/checkpoint/kd-ced-tiny-2026-10-29-18-20-51/student/last_best_checkpoint.pt \
  --high-ckpt /marimo/output/high_expert/last_best_checkpoint.pt \
  --teacher-fold0 /marimo/pretrained/beats_teacher_fold0_weights.pt \
  --teacher-ema /marimo/pretrained/beats_teacher_ema_standalone.pt \
  --data-root /marimo/dataset/mix-dataset-v2 \
  --cache-dir /marimo/dataset/cache \
  --split validation
```

### Sử dụng trực tiếp trong mã nguồn Python:
```python
import torch
from src.pipeline import load_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = load_pipeline(
    config_path="config/moe.yaml",
    low_ckpt="/marimo/output/kd-v5/checkpoint/.../last_best_checkpoint.pt",
    high_ckpt="/marimo/output/high_expert/last_best_checkpoint.pt",
    t_fold0_ckpt="/marimo/pretrained/beats_teacher_fold0_weights.pt",
    t_ema_ckpt="/marimo/pretrained/beats_teacher_ema_standalone.pt",
    device=device,
)

# Dự đoán xác suất cho batch âm thanh [B, 64000]
probs = pipeline.predict_proba(wave, snrs_db=snrs)
predictions = pipeline.predict(wave, snrs_db=snrs)
```
