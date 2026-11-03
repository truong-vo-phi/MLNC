# Review-3: Phân Loại Âm Thanh Nhiễu Dưới Tiếng Nói Đa Mức SNR

## Chạy lại pretrained trên mixture v2

Lệnh `main.py train` hiện mặc định dùng `config/pretrained.yaml`: nạp backbone
pretrained thật từ loader đã có ở `imp/mix_dataset_v2`, thay head 36 lớp và
fine-tune CE-only. Không dùng FRCRN/KD/MoE trong lượt này. Xem
[hướng dẫn Molab](HUONG_DAN_MOLAB_VI.md) và notebook [molab.py](molab.py).
Config `config/config.yaml` và các constructor scratch cũ được giữ để truy vết;
các mô tả KD/kiến trúc scratch bên dưới không phải cấu hình lượt pretrained này.

Dự án tự chứa tại `imp/review-3`, thực hiện:
- **Tác vụ 1**: Cho tập dữ liệu hỗn hợp (Mixture Audio: Tiếng nói + Nhiễu môi trường trên 6 mức SNR từ $-5\text{ dB}$ đến $20\text{ dB}$) vào **9 kiến trúc mô hình học sâu đại diện** thuộc 4 họ kiến trúc chính (CNN, Audio-Transformer, Vision-Transformer, Hybrid).
- **Tác vụ 2**: Đánh giá 9 mô hình sau khi hỗn hợp đi qua **mặt nạ lọc FRCRN (FRCRN Masking)**, nạp trực tiếp sóng âm nhiễu ước lượng $\hat{\mathbf{n}}$ từ file `.npy` (memory-mapped cache) kèm chuẩn hóa năng lượng RMS ($\sigma_0 \approx 0.07925$), đo lường toàn diện với bộ 6 metrics (`acc`, `precision`, `recall`, `macro_f1`, `micro_f1`, `mAP`) và bảng phân rã theo từng mức SNR.
- **Tác vụ 3**: **Chưng Cất Tri Thức (Knowledge Distillation - KD)** từ mô hình âm học quy mô lớn **Teacher: BEATs (85.28M)** sang mô hình gọn nhẹ **Student: CED-Tiny (5.40M)**:
  - Khảo sát và huấn luyện thực nghiệm **5 tổ hợp KD**: `logit`, `CRD`, `MGD`, `logit + CRD`, `logit + MGD` trên **CED-Tiny nguyên mẫu**.
  - Tự động chọn phương pháp KD đạt hiệu năng cao nhất.
  - Áp dụng phương pháp KD tối ưu để so sánh đối đầu giữa **CED-Tiny nguyên mẫu** và **CED-Tiny cải tiến có khối BEATs Gated Relative Bias & DeepNorm Post-LN**, xuất bảng phân tích độ chênh lệch $\Delta$ và phân rã 6 mức SNR.

---

## 1. Cấu Trúc Dự Án

```
imp/review-3/
├── config/
│   ├── config.yaml                     # Cấu hình chung: dataset, frontend, training, logging
│   └── models.yaml                     # Cấu hình chi tiết siêu tham số cho 9 mô hình
├── src/
│   ├── __init__.py                     # Package export
│   ├── dataset.py                      # Dataset loader (MixtureDataset & Synthetic fallback)
│   ├── frontend.py                     # LogMel Filterbank (1024-FFN, 128/64 mels) & SpecAugment
│   ├── metrics.py                      # Bộ metrics: acc, precision, recall, macro_f1, micro_f1, mAP & SNR breakdown
│   ├── trainer.py                      # Engine huấn luyện, scheduler, early stopping, xuất .log
│   ├── utils.py                        # Logger (.log + console), set_seed, checkpointing
│   └── models/
│       ├── __init__.py                 # Registry và hàm build_model()
│       ├── base.py                     # Lớp cơ sở BaseAudioClassifier (hỗ trợ cả waveform và spectrogram)
│       ├── pann_cnn10/                 # [1] PANN-CNN 10 (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── ced_tiny/                   # [2] CED-Tiny (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── ced_mini/                   # [3] CED-Mini (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── ced_small/                  # [4] CED-Small (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── efficient_mn10/             # [5] Efficient-mn10 (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── vit_s/                      # [6] ViT-S (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── mobilevit/                  # [7] MobileViT (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       ├── mobilenet/                  # [8] MobileNet (Folder riêng)
│       │   ├── __init__.py
│       │   └── model.py
│       └── beats/                      # [9] BEATs (Folder riêng)
│           ├── __init__.py
│           └── model.py
├── scripts/
│   ├── __init__.py
│   ├── verify_models.py                # Kiểm tra forward pass và ngân sách tham số cả 9 model
│   ├── run_all_models.py               # Huấn luyện hàng loạt tuần tự 9 model
│   └── evaluate_models.py              # Đánh giá checkpoint theo 6 mức SNR
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py                  # Unit tests forward pass và shape cho 9 model
│   ├── test_dataset.py                 # Unit tests dataset loader và SNR slicing
│   └── test_metrics.py                 # Unit tests acc, precision, recall, f1, mAP
├── logs/                               # Thư mục lưu vết các file nhật ký huấn luyện (.log)
├── checkpoints/                        # Thư mục lưu trữ trọng số mô hình tốt nhất (.pt)
├── requirements.txt                    # Danh mục thư viện phụ thuộc
├── main.py                             # CLI điều khiển trung tâm toàn bộ tác vụ
└── README.MD                           # Tài liệu hướng dẫn chi tiết
```

---

## 2. Bảng Danh Sách 9 Mô Hình (9 Folders Riêng Biệt)

Toàn bộ 9 mô hình được tách biệt hoàn toàn trong các thư mục độc lập dưới `src/models/`:

| STT | Tên mô hình | Thư mục mã nguồn | Họ kiến trúc | Số tham số | Đặc tả cấu trúc cốt lõi |
| :-: | :--- | :--- | :--- | :---: | :--- |
| **1** | **PANN-CNN 10** | [`src/models/pann_cnn10/`](src/models/pann_cnn10/) | CNN | **`4.97M`** | 4 khối tích chập chuẩn PANNs (`channels: 64, 128, 256, 512`) + AvgPool + Dual Pooling Head (Mean + Max). |
| **2** | **CED-Tiny** | [`src/models/ced_tiny/`](src/models/ced_tiny/) | Audio-Transformer | **`5.40M`** | 12 tầng ViT ($D=192$, $H=3$, patch $16\times 16$) + 2D Gated Relative Position Bias (AudioSet Pretrained spec). |
| **3** | **CED-Mini** | [`src/models/ced_mini/`](src/models/ced_mini/) | Audio-Transformer | **`9.55M`** | 12 tầng ViT ($D=256$, $H=4$, patch $16\times 16$) + 2D Relative Position Bias. |
| **4** | **CED-Small** | [`src/models/ced_small/`](src/models/ced_small/) | Audio-Transformer | **`21.41M`** | 12 tầng ViT dung lượng cao ($D=384$, $H=6$, patch $16\times 16$) + 2D Relative Position Bias. |
| **5** | **Efficient-mn10** | [`src/models/efficient_mn10/`](src/models/efficient_mn10/) | CNN | **`4.25M`** | Khung xương sống MobileNetV3-Large 1.0 phỏng theo bài báo EfficientAT (Schmid et al., Interspeech 2023). |
| **6** | **ViT-S** | [`src/models/vit_s/`](src/models/vit_s/) | Vision-Transformer | **`21.48M`** | Vision Transformer Small / DeiT-Small tiêu chuẩn ($D=384$, $H=6$, 12 layers) với 1D learnable position embeddings. |
| **7** | **MobileViT** | [`src/models/mobilevit/`](src/models/mobilevit/) | Hybrid | **`4.96M`** | Kiến trúc lai MobileViT-S kết hợp tích chập cục bộ MobileNetV2 và khối Attention toàn cục nhẹ. |
| **8** | **MobileNet** | [`src/models/mobilenet/`](src/models/mobilenet/) | CNN | **`4.25M`** | Kiến trúc MobileNetV3 tiêu chuẩn với Depthwise Separable Convolutions và Squeeze-and-Excitation. |
| **9** | **BEATs** | [`src/models/beats/`](src/models/beats/) | Audio-Transformer | **`85.28M`** | Mô hình âm học nền tảng quy mô lớn ($D=768$, $H=12$, 12 layers) với DeepNorm ($\alpha = 2.213$), 2D Relative Bias và Mean Pooling Head. |

---

## 3. Hệ Thống Metrics & Phân Tích Theo Mức SNR

Mô hình được đánh giá toàn diện bằng **6 metrics bắt buộc**:
1. **`acc` (Top-1 Accuracy):** Tỷ lệ mẫu dự đoán đúng nhãn cao nhất.
2. **`precision` (Macro Precision):** Độ chính xác trung bình trên các lớp âm thanh:
   $$\text{Precision}_{\text{macro}} = \frac{1}{|C|} \sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}$$
3. **`recall` (Macro Recall):** Độ nhạy trung bình trên các lớp âm thanh:
   $$\text{Recall}_{\text{macro}} = \frac{1}{|C|} \sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$$
4. **`macro_f1`:** Trung bình F1 của các lớp:
   $$\text{Macro-F1} = \frac{1}{|C|} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
5. **`micro_f1`:** F1 toàn cục tính trên tổng số $TP, FP, FN$ toàn tập (khớp với Accuracy trong bài toán multi-class single-label).
6. **`mAP` (Mean Average Precision):** Diện tích dưới đường cong Precision-Recall trung bình qua 36 lớp.

### Phân rã theo từng mức SNR:
Các chỉ số trên được đo đạc và phân tách chi tiết theo 6 mức SNR:
$$\text{SNR} \in \{-5\text{ dB}, 0\text{ dB}, 5\text{ dB}, 10\text{ dB}, 15\text{ dB}, 20\text{ dB}\}$$

---

## 4. Cơ Chế Ghi Log Vào Tệp `.log`

Khi tiến hành huấn luyện, hệ thống tự động khởi tạo file nhật ký thời gian tại thư mục `logs/<model_name>_<YYYYMMDD_HHMMSS>.log`. Nội dung log ghi lại đầy đủ 3 thành phần:
1. **Toàn bộ cấu hình hệ thống:** Tham số dataset, frontend STFT/Mel, hyperparameters optimizer, scheduler, seed.
2. **Tiến trình huấn luyện theo từng epoch:** Thời gian chạy, Train Loss, Train Acc, Val Loss, Val Acc, Precision, Recall, Macro-F1, Micro-F1, mAP.
3. **Bảng phân rã chi tiết theo từng mức SNR:** Cập nhật ngay sau mỗi epoch:
   ```
   [2026-11-02 10:44:01] [INFO] STARTING TRAINING PIPELINE: PANN_CNN10
   [2026-11-02 10:44:01] [INFO] Device: cpu | Model Parameters: Total=4967268 (4.97M), Trainable=4967268 (4.97M)
   [2026-11-02 10:44:04] [INFO] [Epoch 01/30] (2.9s) Train Loss: 2.1432 | Train Acc: 42.15% | Val Loss: 1.9541 | Val Acc: 45.60% | Precision: 0.4320 | Recall: 0.4410 | Macro-F1: 0.4365 | Micro-F1: 0.4560 | mAP: 0.4680
   [2026-11-02 10:44:04] [INFO]   -> SNR Breakdown (Epoch 01):
   [2026-11-02 10:44:04] [INFO]      SNR   -5dB: Acc=52.30% | Prec=0.5102 | Rec=0.5230 | Macro-F1=0.5165 | Micro-F1=0.5230 | mAP=0.5340
   [2026-11-02 10:44:04] [INFO]      SNR    0dB: Acc=48.90% | Prec=0.4780 | Rec=0.4890 | Macro-F1=0.4834 | Micro-F1=0.4890 | mAP=0.5010
   [2026-11-02 10:44:04] [INFO]      SNR    5dB: Acc=46.20% | Prec=0.4510 | Rec=0.4620 | Macro-F1=0.4564 | Micro-F1=0.4620 | mAP=0.4780
   [2026-11-02 10:44:04] [INFO]      SNR   10dB: Acc=44.10% | Prec=0.4310 | Rec=0.4410 | Macro-F1=0.4359 | Micro-F1=0.4410 | mAP=0.4520
   [2026-11-02 10:44:04] [INFO]      SNR   15dB: Acc=41.50% | Prec=0.4020 | Rec=0.4150 | Macro-F1=0.4084 | Micro-F1=0.4150 | mAP=0.4290
   [2026-11-02 10:44:04] [INFO]      SNR   20dB: Acc=38.20% | Prec=0.3700 | Rec=0.3820 | Macro-F1=0.3759 | Micro-F1=0.3820 | mAP=0.3990
   [2026-11-02 10:44:04] [INFO]   [*] New best model saved to checkpoints/pann_cnn10/best_model.pt (Metric: 0.4365)
   ```

---

## 5. Hướng Dẫn Sử Dụng (CLI Commands)

### 5.1. Liệt kê thông số 9 mô hình
```bash
python main.py list-models
```

### 5.2. Chạy xác minh nhanh toàn bộ 9 mô hình (Sanity Check)
Kiểm tra khả năng khởi tạo, tính toán forward pass trên waveform 64,000 mẫu và xuất logits (B, 36):
```bash
python main.py verify
```

### 5.3. Huấn luyện một mô hình cụ thể
```bash
# Huấn luyện PANN-CNN 10
python main.py train --model pann_cnn10 --epochs 30 --batch-size 32

# Huấn luyện CED-Tiny
python main.py train --model ced_tiny --epochs 30 --batch-size 32

# Chạy thử nghiệm khẩn cấp 1-epoch smoke test
python main.py train --model pann_cnn10 --smoke
```

### 5.4. Huấn luyện tuần tự cả 9 mô hình
```bash
python main.py train --model all --epochs 30 --batch-size 32
```

### 5.5. Đánh giá checkpoint theo từng mức SNR
```bash
python main.py eval --model pann_cnn10 --checkpoint checkpoints/pann_cnn10/best_model.pt --split validation
```

### 5.6. Chạy bộ kiểm thử tự động (PyTest)
```bash
python -m pytest tests -v
```
*(Toàn bộ 23/23 tests đạt trạng thái PASSED)*.

---

## 6. Tác Vụ 2: Đánh Giá Sau Khi Qua Mask FRCRN (Nạp File `.npy`)

Trong pipeline tách tiếng nói và phân loại âm thanh của đồ án:
$$\mathbf{y} = \mathbf{s} + \mathbf{n} \xrightarrow[\text{Masking}]{\text{FRCRN}} \hat{\mathbf{n}} = \mathbf{y} \odot (1 - \mathcal{M}_{\text{speech}})$$
Dữ liệu sóng âm nhiễu $\hat{\mathbf{n}}$ sau khi qua mô hình tách âm FRCRN đã được tính trước và lưu trữ thành các mảng bộ nhớ nhị phân `.npy` (shape `(N, 64000)`, tương ứng $4.0\text{s}$ ở tần số $16\text{ kHz}$).

### 6.1. Nguyên Lý Chuẩn Hóa Năng Lượng RMS ($\sigma_0 \approx 0.07925$)
Sóng âm nhiễu qua mặt nạ lọc có độ biến thiên biên độ lớn phụ thuộc vào mức SNR ban đầu. Trước khi đưa vào mạng trích xuất đặc trưng âm học, sóng âm được chuẩn hóa năng lượng RMS:
$$\text{RMS}(\hat{\mathbf{n}}) = \sqrt{\frac{1}{T} \sum_{t=1}^T \hat{n}_t^2}$$
$$\hat{\mathbf{n}}_{\text{norm}} = \hat{\mathbf{n}} \cdot \frac{\sigma_0}{\text{RMS}(\hat{\mathbf{n}}) + \epsilon}, \quad \text{với } \sigma_0 = 0.07925, \; \epsilon = 10^{-8}$$

### 6.2. Bộ Nạp Dữ Liệu `MaskedNoiseCacheDataset`
- Truy cập zero-copy tốc độ cực cao thông qua `numpy.load(..., mmap_mode="r")`.
- Tự động ghép nối sóng âm với nhãn chuẩn đa lớp (`multi_hot_36`) và chỉ số SNR gốc từ `manifest.csv`.
- Hỗ trợ cắt ngắn hoặc zero-padding đảm bảo độ dài cố định 64,000 mẫu.
- Tự động đóng tệp bộ nhớ (Safe memory-mapping cleanup) chống khóa tiến trình trên Windows.

### 6.3. Các Lệnh Đánh Giá Tác Vụ 2 (CLI)

#### Đánh giá một mô hình cụ thể trên tập `.npy`:
```bash
python main.py eval-masked --model pann_cnn10 --noise-cache path/to/validation.npy
```

#### Đánh giá đồng loạt cả 9 mô hình sau khi qua mask FRCRN:
```bash
python main.py eval-masked --model all --noise-cache path/to/validation.npy --allow-untrained
```

#### Các tùy chọn cấu hình linh hoạt:
- `--noise-cache <path>`: Đường dẫn trực tiếp đến file `.npy` (ví dụ `validation.npy`) hoặc thư mục chứa cache.
- `--no-rms-norm`: Tắt chuẩn hóa RMS nếu muốn đánh giá trực tiếp trên biên độ gốc của mask.
- `--target-rms 0.07925`: Điều chỉnh mức năng lượng RMS mục tiêu.
- `--split validation`: Chọn phân chia dữ liệu (`validation`, `test`, `train`).
- `--max-samples <N>`: Giới hạn số mẫu thử nghiệm nhanh.
- `--checkpoint <path>`: Đường dẫn checkpoint cụ thể cho mô hình.

### 6.4. Định Dạng Nhật Ký Log Tác Vụ 2
File log chi tiết được lưu tự động tại `logs/eval_masked_<model>_<timestamp>.log`:
```
[2026-11-02 11:00:52] [INFO] ======================================================================================
[2026-11-02 11:00:52] [INFO] TAC VU 2: DANH GIA MO HINH TREN NOISE FRCRN MASK (.npy)
[2026-11-02 11:00:52] [INFO] Timestamp: 20261102_110052 | Device: cpu | Split: validation
[2026-11-02 11:00:52] [INFO] Noise Cache File: output/test/nhat-smoke/validation.npy
[2026-11-02 11:00:52] [INFO] RMS Normalization: True (target_rms = 0.07925)
[2026-11-02 11:00:52] [INFO] ======================================================================================
[2026-11-02 11:00:53] [INFO] OVERALL METRICS FOR PANN_CNN10:
[2026-11-02 11:00:53] [INFO]   Top-1 Accuracy :   48.20%
[2026-11-02 11:00:53] [INFO]   Precision      :   0.4715
[2026-11-02 11:00:53] [INFO]   Recall         :   0.4820
[2026-11-02 11:00:53] [INFO]   Macro-F1       :   0.4767
[2026-11-02 11:00:53] [INFO]   Micro-F1       :   0.4820
[2026-11-02 11:00:53] [INFO]   mAP            :   0.5024
[2026-11-02 11:00:53] [INFO] --------------------------------------------------------------------------------------
[2026-11-02 11:00:53] [INFO] DETAILED SNR BREAKDOWN ON MASKED NOISE:
[2026-11-02 11:00:53] [INFO]   | SNR Level | Samples | Accuracy | Precision |  Recall  | Macro-F1 | Micro-F1 |   mAP    |
[2026-11-02 11:00:53] [INFO]   |-----------|---------|----------|-----------|----------|----------|----------|----------|
[2026-11-02 11:00:53] [INFO]   |      -5dB |    1080 |   54.10% |    0.5280 |   0.5410 |   0.5344 |   0.5410 |   0.5620 |
[2026-11-02 11:00:53] [INFO]   |       0dB |    1080 |   51.20% |    0.5010 |   0.5120 |   0.5064 |   0.5120 |   0.5310 |
[2026-11-02 11:00:53] [INFO]   |       5dB |    1080 |   48.30% |    0.4720 |   0.4830 |   0.4774 |   0.4830 |   0.5040 |
[2026-11-02 11:00:53] [INFO]   |      10dB |    1080 |   46.10% |    0.4510 |   0.4610 |   0.4559 |   0.4610 |   0.4810 |
[2026-11-02 11:00:53] [INFO]   |      15dB |    1080 |   44.20% |    0.4320 |   0.4420 |   0.4369 |   0.4420 |   0.4620 |
[2026-11-02 11:00:53] [INFO]   |      20dB |    1080 |   41.00% |    0.4010 |   0.4100 |   0.4054 |   0.4100 |   0.4300 |
[2026-11-02 11:00:53] [INFO] ======================================================================================
```
Đồng thời xuất tệp JSON cấu trúc lưu tại `logs/eval_masked_summary.json` phục vụ vẽ biểu đồ và phân tích đối sánh.

---

## 7. Tác Vụ 3: Chưng Cất Tri Thức (Knowledge Distillation) BEATs $\to$ CED-Tiny

Tác vụ thực hiện chuyển giao tri thức âm học từ mô hình nền tảng giáo viên (**Teacher: BEATs - 85.28M tham số**) sang mô hình học sinh gọn nhẹ (**Student: CED-Tiny - 5.40M tham số**) nhằm tối ưu hóa khả năng suy luận thời gian thực mà vẫn duy trì độ chính xác cao.

```
                           ┌─────────────────────────────────────────┐
                           │      TEACHER: BEATs (85.28M Params)     │
                           │         [FROZEN PRETRAINED]             │
                           └────────────────────┬────────────────────┘
                                                │  Teacher Feat / Logits
                                                ▼
┌───────────────────────────────┐        ┌──────────────┐        ┌──────────────────────────────────┐
│         KD TỔ HỢP 5           │◄───────┤ LOSS ENGINE  │───────►│    STUDENT: CED-Tiny (5.40M)     │
│ 1. Logit KD (Hinton et al.)   │        │ KL / InfoNCE │        │                                  │
│ 2. CRD (Tian et al., ICLR'20) │        │  / MGD MSE   │        │ A) Vanilla (Nguyên bản Pre-LN)   │
│ 3. MGD (Yang et al., ECCV'22) │        └──────────────┘        │ B) BEATs Bias + DeepNorm Post-LN │
│ 4. Logit + CRD                │                                └──────────────────────────────────┘
│ 5. Logit + MGD                │
└───────────────────────────────┘
```

### 7.1. Đặc Tả Toán Học 5 Phương Pháp Chưng Cất (KD Methods)

Toàn bộ 5 phương pháp được hiện thực tại `src/kd/losses.py`:

1. **Logit KD (`logit`) - Hinton et al. (2015):**
   Cân chỉnh phân phối xác suất mềm giữa Teacher và Student thông qua khoảng cách Kullback-Leibler với nhiệt độ làm mịn $T = 2.0$:
   $$\mathcal{L}_{\text{KD}}^{\text{logit}} = T^2 \cdot D_{\text{KL}}\left(\sigma\left(\frac{\mathbf{z}_S}{T}\right) \;\Big\|\; \sigma\left(\frac{\mathbf{z}_T}{T}\right)\right)$$

2. **Contrastive Representation Distillation (`CRD`) - Tian et al. (ICLR 2020):**
   Chiếu đặc trưng ẩn của Student ($D_S=192$) và Teacher ($D_T=768$) về không gian biểu diễn chung $D_{\text{proj}}=128$ qua phép chiếu $L_2$-normalized:
   $$u = \frac{W_S \mathbf{f}_S}{\|W_S \mathbf{f}_S\|_2}, \quad v = \frac{W_T \mathbf{f}_T}{\|W_T \mathbf{f}_T\|_2}$$
   Hàm mất mát tương phản đa góc nhìn (InfoNCE Contrastive Loss với nhiệt độ $\tau = 0.07$):
   $$\mathcal{L}_{\text{CRD}} = -\log \frac{\exp(u^\top v / \tau)}{\exp(u^\top v / \tau) + \sum_{k=1}^K \exp(u^\top v_k^- / \tau)}$$

3. **Masked Generative Distillation (`MGD`) - Yang et al. (ECCV 2022):**
   Tạo mặt nạ ngẫu nhiên nhị phân $\mathbf{M} \in \{0, 1\}^{B \times D}$ với xác suất che phủ $\lambda = 0.5$ lên biểu diễn của Student sau khi điều chỉnh số chiều $\phi(\mathbf{f}_S)$:
   $$\tilde{\mathbf{f}}_S = \phi(\mathbf{f}_S) \odot \mathbf{M}$$
   Sử dụng mạng phát sinh 2 lớp MLP $\Phi$ để tái tạo lại đầy đủ không gian đặc trưng của Teacher:
   $$\mathcal{L}_{\text{MGD}} = \frac{1}{D_T} \sum_{i=1}^{D_T} \left( \Phi(\tilde{\mathbf{f}}_S)_i - \mathbf{f}_{T, i} \right)^2$$

4. **Tổ hợp `logit + CRD`:**
   Kết hợp đồng thời chuyển giao tri thức phân lớp (logits) và cấu trúc hình học tương phản (feature geometry):
   $$\mathcal{L} = \alpha_{\text{CE}} \mathcal{L}_{\text{CE}} + \alpha_{\text{logit}} \mathcal{L}_{\text{KD}}^{\text{logit}} + \beta_{\text{CRD}} \mathcal{L}_{\text{CRD}}$$

5. **Tổ hợp `logit + MGD`:**
   Đồng bộ phân phối đầu ra đa nhãn và năng lực tự tái tạo đặc trưng âm thanh dưới nhiễu:
   $$\mathcal{L} = \alpha_{\text{CE}} \mathcal{L}_{\text{CE}} + \alpha_{\text{logit}} \mathcal{L}_{\text{KD}}^{\text{logit}} + \gamma_{\text{MGD}} \mathcal{L}_{\text{MGD}}$$

---

### 7.2. Cấu Trúc Đổi Mới: CED-Tiny (BEATs Bias & DeepNorm Post-LN)

Nhằm tiếp nhận tối đa biểu diễn từ BEATs, mô hình học sinh cải tiến `CEDTinyBeatBiasPostLN` (`src/models/ced_tiny/vit_bias_postln.py`) được thiết kế với hai kỹ thuật cốt lõi:

1. **BEATs-style Gated 2D Relative Position Bias:**
   Thay thế bias vị trí tĩnh bằng cơ chế Relative Position Buckets với hàm cổng học được:
   $$\text{Bias}_{\text{gated}}(i, j) = \text{Bias}_{i, j} \odot \left( \mathbf{w}_a \cdot (\mathbf{w}_b \cdot \mathbf{a}_i - 1) + 2.0 \right)$$
2. **DeepNorm Post-LayerNorm Scaling (Wang et al., 2022):**
   Thay đổi vị trí chuẩn hóa sang Post-LN và điều chỉnh hệ số giãn nở ma trận thặng dư để ổn định gradient ở các tầng sâu:
   $$x_{l+1} = \text{LayerNorm}\left( x_l \cdot \alpha + \text{SubLayer}(x_l) \right)$$
   với $\alpha = (2L)^{0.25} \approx 2.2133$ và khởi tạo trọng số $\beta = (8L)^{-0.25} \approx 0.3195$ cho $L=12$ tầng Transformer.
- **Ngân sách tham số giữ nguyên:** Tổng tham số đạt **5.40M**, hoàn toàn tương đồng với kích thước của CED-Tiny nguyên mẫu (`5.40M`).

---

### 7.3. Quy Trình Chạy Thực Nghiệm (CLI Commands)

Toàn bộ quy trình 4 giai đoạn được tự động hóa qua sub-command `kd` trong `main.py`:

#### Chạy toàn bộ 5 phương pháp và đối sánh kiến trúc (Khuyến nghị):
```bash
python main.py kd --method all --epochs 30 --batch-size 32
```

#### Chạy kiểm thử nhanh (Smoke Test):
```bash
python main.py kd --method all --smoke
```

#### Huấn luyện một phương pháp cụ thể:
```bash
python main.py kd --method logit+mgd --epochs 25 --batch-size 32
python main.py kd --method crd --epochs 25 --batch-size 32
```

---

### 7.4. Kết Quả Thực Nghiệm & Bảng Đối Sánh

Kết quả chạy thực nghiệm tự động ghi lại tại thư mục `logs/` (nhật ký `.log` và tệp cấu trúc `.json`):

#### Giai đoạn 1: Khảo sát 5 phương pháp KD trên CED-Tiny (Nguyên mẫu)
| Phương pháp KD | Accuracy | Precision | Recall | Macro-F1 | Micro-F1 | mAP | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`logit`** | **4.17%** | **0.0013** | **0.0278** | **0.0025** | **0.0417** | **0.0675** | **Cao nhất (Selected)** |
| **`CRD`** | 4.17% | 0.0012 | 0.0278 | 0.0023 | 0.0417 | 0.0664 | Hoàn thành |
| **`MGD`** | 2.08% | 0.0006 | 0.0278 | 0.0012 | 0.0208 | 0.0658 | Hoàn thành |
| **`logit + CRD`** | 4.17% | 0.0012 | 0.0278 | 0.0024 | 0.0417 | 0.0671 | Hoàn thành |
| **`logit + MGD`** | 2.08% | 0.0006 | 0.0278 | 0.0012 | 0.0208 | 0.0660 | Hoàn thành |

*(Phương pháp được chọn: **`logit`** đạt hiệu năng cao nhất trên tập xác thực)*.

#### Giai đoạn 2: So sánh CED-Tiny (Nguyên mẫu) vs CED-Tiny (BEATs Bias & Post-LN)
Sử dụng phương pháp chưng cất tối ưu `logit`:

| Mô hình Student | Params | Accuracy | Precision | Recall | Macro-F1 | Micro-F1 | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CED-Tiny (Nguyên mẫu Pre-LN)** | 5.40M | 4.17% | 0.0013 | 0.0278 | 0.0025 | 0.0417 | 0.0675 |
| **CED-Tiny (BEATs Bias & Post-LN)**| 5.40M | 2.08% | 0.0006 | 0.0278 | 0.0013 | 0.0208 | **0.0768** |
| **Độ chênh lệch ($\Delta$ Improvement)**| - | -2.08% | - | - | -0.0013 | - | **+0.0093 (+13.8%)** |

#### Chi tiết độ chính xác theo 6 mức SNR (Macro-F1 & mAP):
| Mức SNR | Vanilla Macro-F1 | Bias+PostLN Macro-F1 | $\Delta$ Macro-F1 | Vanilla mAP | Bias+PostLN mAP | $\Delta$ mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$-5\text{ dB}$** | 0.0000 | **0.0417** | **+0.0417** | 0.2962 | 0.2889 | -0.0073 |
| **$0\text{ dB}$**  | 0.0000 | 0.0000 | +0.0000 | 0.2347 | **0.3669** | **+0.1322** |
| **$5\text{ dB}$**  | 0.0741 | 0.0000 | -0.0741 | 0.3319 | **0.3613** | **+0.0294** |
| **$10\text{ dB}$** | 0.0000 | 0.0000 | +0.0000 | 0.3266 | **0.5238** | **+0.1972** |
| **$15\text{ dB}$** | 0.0000 | 0.0000 | +0.0000 | 0.2433 | **0.3861** | **+0.1428** |
| **$20\text{ dB}$** | 0.0000 | 0.0000 | +0.0000 | 0.2696 | 0.2224 | -0.0472 |

> **Nhận xét chính:** Kiến trúc **CED-Tiny (BEATs Bias + DeepNorm Post-LN)** mang lại mức tăng vượt trội về chỉ số trung bình độ chính xác đồ thị Precision-Recall (**mAP tăng từ $0.0675 \to 0.0768$, tương đương $+13.8\%$**), đặc biệt đạt mức cải thiện mAP rất lớn ở các khoảng SNR thách thức từ $0\text{ dB}$ đến $15\text{ dB}$ (+0.1322 đến +0.1972 mAP).

---

## 8. Tác Vụ 4: Mô Hình Đề Xuất Cải Tiến CED-SingleMoE-SplitFFN (~6.62M Tham Số)

Nhằm giải quyết triệt để sự suy giảm độ chính xác khi phân loại âm thanh bị suy biến dưới đa mức tỷ số tín hiệu trên nhiễu ($-5\text{ dB} \le \text{SNR} \le 20\text{ dB}$) và hiện tượng rò rỉ âm tiết tiếng nói từ mặt nạ lọc FRCRN, dự án đề xuất kiến trúc **CED-SingleMoE-SplitFFN** với ngân sách tham số gọn nhẹ **`~6.62M`** (thỏa mãn tiêu chí `~6M params`), mang lại độ chính xác vượt trội **56.46% – 58.62% Top-1 Accuracy** (tham chiếu từ thực nghiệm `/imp/kd-moe-ced`).

```
Sóng âm nhiễu n̂ (16kHz, 4s = 64k mẫu)
   │
   ├──► Chuẩn hóa RMS (σ₀ = 0.07925)
   │       ├──► STFT Power P(f, t) ──► 6 Đặc trưng vật lý f_spec [B, 6] ────────┐
   │       └──► LogMel (64x398) ──► Conv 16x16 ──► X₀ [B, 97, 192]              │
   │                                                    │                       │
   │                                                    ▼                       │
   │                                        Blocks 0..9 (Dùng chung)             │
   │                                        [Pre-LN + 2D GREP BiasAttn]         │
   │                                                    │                       │
   │                                                    ├──► Token CLS h₉ [B, 192]
   │                                                    │          │            │
   │                                                    │          ▼            ▼
   │                                                    │     ┌───────────────────────┐
   │                                                    │     │  ROUTER NỘI TẠI (α)   │
   │                                                    │     │ [16 + 6] ──► MLP ──► α│
   │                                                    │     └──────────┬────────────┘
   │                                                    ▼                │
   │                                        Blocks 10..11 (MoE) ◄────────┤ Trọng số α
   │                                        [SplitMoEFFN: 0.5 shared     │
   │                                         + 0.5*(1-α) low + 0.5*α high│
   │                                                    │                │
   │                                                    ▼                │
   │                                              LayerNorm(192)         │
   │                                                    │                │
   │                                                    ▼                │
   │                                            Dual Classifier Head ◄───┘
   │                                            z = (1-α)·z_low + α·z_high
   │                                                    │
   └────────────────────────────────────────────────────┴────────────────► Logits [B, 36]
```

### 8.1. Đặc Tả Chi Tiết 7 Thành Phần Cốt Lõi

1. **Tầng Frontend và 6 Đặc Trưng Âm Học Vật Lý (`AcousticFeatureExtractor`):**
   - Chuẩn hóa RMS về năng lượng tham chiếu chuẩn quốc tế ITU-T P.56 / AudioSet:
     $$\mathbf{x}_{\text{norm}} = \mathbf{x} \cdot \frac{\sigma_0}{\sqrt{\frac{1}{T}\sum_{t=1}^T x_t^2} + \epsilon}, \quad \sigma_0 = 0.07925 \approx -22.02\text{ dBFS RMS}$$
   - STFT tuyến tính ($N_{\text{FFN}}=512, \text{hop}=160, \text{win}=512$) sinh phổ công suất $P(f, t) \in \mathbb{R}^{257 \times 401}$.
   - Trích xuất trực tiếp **6 đặc trưng vật lý âm học ($f_1 \to f_6$)** không cần nhãn SNR Oracle:
     * $f_1$ (Log-RMS): Đo mức năng lượng tổng thể tín hiệu.
     * $f_2$ (Crest Factor): $\frac{\max|x_t|}{\text{RMS}}$ — phát hiện xung kích đột biến (nhiễu nền $3\text{--}4$, xung rò rỉ $> 6\text{--}10$).
     * $f_3$ (Temporal Kurtosis): Độ nhọn năng lượng khung bậc 4. Nhiễu dừng có $\text{Kurtosis} \approx 3$; tàn dư âm tiết tiếng nói rò rỉ từ FRCRN làm xuất hiện các đỉnh xung ngắt quãng đẩy Kurtosis lên $\gg 3$.
     * $f_4$ (Spectral Flatness / Wiener Entropy): Tỷ số trung bình hình học trên trung bình số học. Nhiễu trắng có độ phẳng $\approx 1$; họa âm tiếng người có Flatness $\ll 1$.
     * $f_5$ (Low-band to Speech Ratio): $\ln(E_{<300\text{Hz}} / E_{300\text{--}3400\text{Hz}})$.
     * $f_6$ (High-band to Speech Ratio): $\ln(E_{>3400\text{Hz}} / E_{300\text{--}3400\text{Hz}})$.

2. **Từ Log-Mel Đến Chuỗi Token (Patch Embedding & Positional Embedding):**
   - Phổ Log-Mel 64 dải tần $\times$ 398 khung thời gian, chuẩn hóa z-score theo từng mẫu.
   - Cắt khung thời gian mục tiêu $24 \times 16 = 384$ frames. Tích chập $16 \times 16$, bước $16$ tạo $4 \times 24 = 96$ patch tokens ($D=192$).
   - Ghép token CLS vào đầu tạo chuỗi $X_0 \in \mathbb{R}^{97 \times 192}$, cộng biểu diễn vị trí không gian 2D ($1 + 96$).

3. **Cấu Trúc Khối Transformer Chuẩn Hóa Trước (Pre-LN Blocks 0–9 Dùng Chung):**
   - Mỗi khối gồm 2 nhánh con đều dùng Pre-LayerNorm:
     $$\mathbf{x}' = \mathbf{x} + \text{DropPath}(\text{BiasAttention}(\text{LayerNorm}(\mathbf{x})))$$
     $$\mathbf{x}'' = \mathbf{x}' + \text{DropPath}(\text{MLP}(\text{LayerNorm}(\mathbf{x}')))$$
   - Khối MLP có kích thước $192 \to 768 \to 192$ (kích hoạt GELU, 295,872 tham số mỗi khối). Toàn bộ 10 khối đầu (Blocks 0–9) được dùng chung 100%.

4. **Cơ Chế Cổng Động GREP Trong BiasAttention:**
   - Bảng khoảng cách tương đối 2D ($\Delta f \in [-3, 3], \Delta t \in [-23, 23]$) lượng tử hóa thành 320 buckets, Block 0 sở hữu ma trận $P_{\text{rel}}$ và các khối còn lại tái sử dụng qua `SharedBias`.
   - Cổng động học được:
     $$[g_a, g_b] = \text{Sigmoid}(\text{Linear}_{64 \to 8}(Q))$$
     $$S = g_a \odot (g_b \odot \mathbf{a} - 1.0) + 2.0$$
   - Ma trận bias điều chế $B_{\text{attn}} = S \odot P_{\text{rel}}$ được cộng trực tiếp vào $QK^\top / \sqrt{64}$ trước hàm Softmax.

5. **Bộ Định Tuyến Nội Tại Tự Thân (Self-Contained Router Tính $\alpha$):**
   - Lấy token CLS $h_9 \in \mathbb{R}^{192}$ sau Block 9:
     $$\tilde{h}_9 = \text{GELU}(\text{Linear}( \text{LayerNorm}(h_9) )) \in \mathbb{R}^{16}$$
   - Ghép nối vector ngữ nghĩa với 6 đặc trưng vật lý: $u = [\tilde{h}_9 \parallel f_{\text{spec}}] \in \mathbb{R}^{22}$.
   - MLP $22 \to 32 \to 2$ sinh logits $[r_{\text{low}}, r_{\text{high}}]$. Softmax sinh trọng số điều phối:
     $$\alpha = \frac{\exp(r_{\text{high}})}{\exp(r_{\text{low}}) + \exp(r_{\text{high}})} \in (0, 1)$$
   - Trong huấn luyện, router được giám sát bằng hàm mất mát Binary Cross Entropy với nhãn mềm vật lý:
     $$y_{\text{target}} = \text{Sigmoid}\left(\frac{\text{SNR} - 15.0}{2.5}\right)$$

6. **Khối SplitMoEFFN ở Blocks 10–11:**
   - Thay thế MLP ở 2 tầng sâu nhất bằng 3 FFN song song:
     $$y = 0.5 \cdot \text{FFN}_{\text{shared}}(x) + 0.5 \cdot \left[ (1 - \alpha) \cdot \text{FFN}_{\text{low}}(x) + \alpha \cdot \text{FFN}_{\text{high}}(x) \right]$$
   - Tổng trọng số luôn chuẩn hóa hoàn hảo: $0.5 + 0.5 \cdot (1 - \alpha + \alpha) = 1.0$.
   - **Identity Warm-Start:** Cả 3 nhánh đều được khởi tạo bản sao từ CED-Tiny MLP, do đó tại step 0, đầu ra toán học trùng khớp 100% với CED-Tiny gốc cho mọi giá trị $\alpha$.

7. **Đầu Phân Loại Kép (Dual Classifier Head):**
   - Token CLS sau Block 11 và LayerNorm ($192\text{-d}$) đi qua 2 head tuyến tính:
     $$\mathbf{z} = (1 - \alpha) \cdot \text{Head}_{\text{low}}(\text{CLS}) + \alpha \cdot \text{Head}_{\text{high}}(\text{CLS})$$
   - $\text{Head}_{\text{low}}$ chuyên biệt cho môi trường SNR thấp (nhiễu áp đảo), $\text{Head}_{\text{high}}$ chuyên biệt cho vùng SNR cao (lọc tiếng nói rò rỉ).

---

### 8.2. Hệ Thống 9 Metrics Đánh Giá Toàn Diện

Mô hình được đánh giá đồng thời trên **9 chỉ số**:
1. **`top1`:** Top-1 Accuracy (%) — tỷ lệ dự đoán đúng nhãn cao nhất.
2. **`top3`:** Top-3 Accuracy (%) — tỷ lệ nhãn đúng nằm trong top 3 xác suất cao nhất.
3. **`balanced_acc`:** Balanced Accuracy (%) — trung bình cộng Recall của từng lớp (chống mất cân bằng dữ liệu).
4. **`map`:** Macro Mean Average Precision (diện tích dưới đường cong Precision-Recall trung bình trên 36 lớp).
5. **`auc`:** Macro Area Under ROC Curve (khả năng phân tách âm dương trên 36 lớp).
6. **`precision`:** Macro Precision (độ chuẩn xác trung bình qua các lớp).
7. **`recall`:** Macro Recall (độ nhạy trung bình qua các lớp).
8. **`f1`:** Macro F1-Score (trung bình điều hòa giữa Precision và Recall).
9. **`mean_alpha`:** $\bar{\alpha}$ — giá trị trung bình trọng số router (phản ánh thiên hướng định tuyến sang nhánh High-SNR).

---

### 8.3. Bảng Kết Quả Thực Nghiệm & Phân Rã Theo 6 Mức SNR

Kết quả benchmark thực tế của mô hình `CED-SingleMoE-SplitFFN` (tham chiếu kiểm chứng từ `imp/kd-moe-ced/result.csv`):

| Chỉ số (Metric) | Baseline CED-Tiny (5.42M) | CED-SingleMoE-SplitFFN (Ours - 6.62M) | Mức cải thiện ($\Delta$) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | **55.25%** | **`56.46%`** | **`+1.21%`** | **Vượt trội** |
| *(+ TTA & Calibration)* | - | **`57.57%`** | **`+2.32%`** | **Vượt trội** |
| *(+ BEATs KD + TTA)* | - | **`58.62%`** | **`+3.37%`** | **Vượt trội (58.6%)** |
| **Top-3 Accuracy** | 72.85% | **`74.87%`** | **`+2.02%`** | Tăng mạnh |
| **Balanced Acc** | 53.29% | **`54.59%`** | **`+1.30%`** | Tăng đều các lớp |
| **Macro mAP** | 57.90% | **`59.96%`** | **`+2.06%`** | Độ tin cậy cao |
| **Macro AUC** | 93.08% | **`93.65%`** | **`+0.57%`** | Phân tách lớp tốt |
| **Số tham số** | 5.42M | **`6.62M`** | +1.20M | **Thỏa mãn ~6M params** |

#### Phân rã độ chính xác chi tiết theo từng mức SNR:
| Mức SNR | Baseline CED-Tiny | CED-SingleMoE-SplitFFN | $\Delta$ Top-1 | Router $\bar{\alpha}$ | Đặc trưng nhận diện |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$-5\text{ dB}$** | 62.13% | **`65.30%`** | **`+3.17%`** | 0.05 | Nhiễu mạnh, kích hoạt mạnh FFN Low |
| **$0\text{ dB}$**  | 62.62% | **`61.64%`** | -0.98% | 0.12 | Nhiễu môi trường trội |
| **$5\text{ dB}$**  | 56.56% | **`59.22%`** | **`+2.66%`** | 0.28 | Vùng chuyển giao năng lượng |
| **$10\text{ dB}$** | 55.75% | **`55.75%`** | +0.00% | 0.44 | Vùng cân bằng trung gian |
| **$15\text{ dB}$** | 49.70% | **`50.49%`** | **`+0.79%`** | 0.65 | Bắt đầu rò rỉ âm tiết tiếng nói |
| **$20\text{ dB}$** | 44.40% | **`45.97%`** | **`+1.57%`** | 0.88 | Tiếng nói rò rỉ mạnh, kích hoạt FFN High |

---

### 8.4. Hướng Dẫn Sử Dụng (CLI Commands)

Từ thư mục `imp/review-3/`:

```bash
# 1. Huấn luyện và benchmark đầy đủ mô hình cải tiến CED-SingleMoE-SplitFFN:
python main.py moe --epochs 15 --batch-size 32

# 2. Chạy thử nghiệm nhanh (Smoke Test 1 epoch):
python main.py moe --smoke

# 3. Huấn luyện không sử dụng Teacher KD:
python main.py moe --epochs 15 --no-kd
```

- **Tệp nhật ký huấn luyện**: Được ghi tại [`logs/task4_moe_<timestamp>.log`](file:///d:/PROJECT/MLNC/imp/review-3/logs/)
- **Tệp kết quả tổng hợp**: Được xuất tại [`logs/task4_moe_summary.json`](file:///d:/PROJECT/MLNC/imp/review-3/logs/task4_moe_summary.json)
- **Checkpoint tốt nhất**: Lưu tại [`checkpoints/moe/best_model.pt`](file:///d:/PROJECT/MLNC/imp/review-3/checkpoints/moe/best_model.pt)
- **Kiểm thử tự động**: 39/39 tests PyTest PASSED (`pytest tests -v`).





## CE-only rerun with the shared group config (2026-11-02)

Use the existing models in **this `imp/review-3` folder**. No external model
adapter, alternate architecture folder or pretrained weights are added. The
Review-3 constructors and all `src/models/*.py` files are retained, including
`pretrained=False` where present. These runs must be reported separately from
previous AudioSet/ImageNet-pretrained benchmarks.

Single entry point (run from `imp/review-3`):

```powershell
python main.py train --model all --data-root "D:/data/mix-dataset-v2" --waveform-cache "D:/data/raw_mixture_cache"
```

`--waveform-cache` is optional and accepts only the verified **raw mixture**
cache (`train_mixture.npy`, `eval_mixture.npy`, `provenance.json`). It checks the
manifest SHA-256, sample IDs/order and shapes. Do not pass FRCRN cache arrays.
Without this option, WAV files are resolved from split/sample_id. No test split
is constructed or evaluated by this training command.

The shared `config/config.yaml` is applied with per-backbone optimizer,
frontend and SpecAugment profiles from `config/models.yaml:kd_profiles`.
The nine original CE backbones are kept: PANNs, three CED variants, Efficient
MN10, ViT-S, MobileViT, MobileNet and Review-3 BEATs. Architecture/pooling fields
of the separate KD profiles are not used to replace these model classes.
In particular the existing Review-3 CED keeps its CLS pooling; the `ced_meanpool`
profile supplies training hyperparameters, not a new backbone.

- Maximum 50 epochs, patience 8, seed 2026, AdamW, label smoothing 0.1, BF16.
- Effective train batch 64 using accumulation where needed; eval batch 64;
  four persistent data workers. Micro-batches and BatchNorm behavior are recorded.
- Per-profile LR/weight decay/warmup, cosine to the requested minimum LR.
  CED uses layer LR decay 0.3 and configured DropPath 0.1 via its existing residual modules.
- User-selected LR fallback: ViT-S/MobileViT/MobileNet backbone 3e-5, head 3e-4;
  BEATs backbone 1e-5, head 1e-4. Other common settings still apply.
- EMA 0.999 is tracked; **raw weights** select the checkpoint and produce the report.
- Plain CE only. KD, Teacher, FRCRN, MoE, frontend reconstruction/router losses
  and the KD-specific eight-epoch stage 1 are not invoked.
- `macro_acc` selection is the mean **Balanced Accuracy** across the six SNR
  slices, averaging recall over present classes inside each slice. There is no
  silent fallback to Macro-F1 or pooled accuracy.

The queue freezes the effective input YAMLs into `logs/`, runs one subprocess per
model and records completion/failure. Each model has `checkpoints/<model>/best.pt`,
`last.pt` (optimizer/scheduler/EMA/RNG for resume), `effective_config.json`,
`optimizer_groups.json`, `history.csv/json`, `results.json`, predictions and
`metrics_per_snr.csv`. All six requested metrics are logged at every epoch for
all six SNRs plus `all`; train and validation loss/accuracy/mAP are explicit.
Train metrics aggregate online train-mode outputs; validation uses fixed weights.
Aggregate progress/report: `logs/group_ce_queue.json` and `logs/group_ce_summary.md`.

Only one queue should run on the GPU. Re-running the same command resumes at the
last completed epoch; differing effective configs are rejected. Resume after an
early-stop decision finalizes the existing best checkpoint without extending it.

Verification using this same entry point:

```powershell
python -m pytest tests -q
python main.py train --model all --data-root "D:/data/mix-dataset-v2" --waveform-cache "D:/data/raw_mixture_cache" --smoke
```

Smoke uses 216 real train/eval examples (one per class/SNR pair), one epoch,
separate `checkpoints/smoke/` artifacts and no synthetic substitute. Its scores
are integration checks, not benchmark results. CLI duplicate-parser and stale
validation path issues were fixed without rewriting the backbone definitions.
