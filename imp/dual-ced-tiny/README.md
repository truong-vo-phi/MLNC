# Dual-CED-Tiny: Hai CED-Tiny với BEATs KD, GREP và Alpha Post-LN

Hệ thống giữ hai backbone CED-Tiny độc lập, với GREP, Alpha Post-LN và learned spectral-token router chuyển từ `imp/moe-v2`. Router lấy sáu đặc trưng waveform cùng CLS token giữa tầng của hai expert; SNR chỉ tạo soft target khi train, không được truyền vào validation/inference. BEATs Fold0/EMA đóng băng và chỉ cung cấp logits cho loss train.

Dự án được triển khai độc lập tại thư mục `imp/dual-ced-tiny`, tuân thủ:
* **Khung sườn kiến trúc & kiểm định:** Chuẩn module hóa của `imp/review-3` (`src/`, `config/`, `scripts/`, `tests/`, `main.py`).
* **Siêu tham số & Dữ liệu:** FRCRN noise cache, RMS 0.07925, effective batch 64, AdamW; `config/config.yaml` đặt tối đa 15 epoch cùng Stage-2 High expert. Với `kd.enabled: true`, loss kết hợp CE và logit KD từ BEATs.

> **Checkpoint:** Router học được tạo thêm tham số, nên checkpoint router ngưỡng trước đây không tương thích strict. Lượt train mới ghi sang `checkpoints_moev2_router/` và log sang `logs_moev2_router/`, giữ nguyên artifact cũ.

---

## 1. Cấu Trúc Thư Mục Dự Án

```
imp/dual-ced-tiny/
├── config/
│   ├── config.yaml                     # Cấu hình chung: dataset, frontend, training, router, logging
│   └── models.yaml                     # Danh mục siêu tham số chi tiết cho Dual-CED-Tiny & Experts
├── src/
│   ├── __init__.py                     # Package export
│   ├── dataset.py                      # Dataset loader (MixtureDataset & Synthetic fallback)
│   ├── frontend.py                     # LogMel Filterbank (64 Mel, 398 frames, 16kHz)
│   ├── metrics.py                      # Bộ 6 metrics (acc, prec, rec, macro_f1, micro_f1, mAP) & SNR breakdown
│   ├── router.py                       # SelfContainedSpectralTokenRouter + legacy EnergySNRRouter
│   ├── trainer.py                      # Engine train, cosine warmup, decoupled LR, .log
│   ├── utils.py                        # Logger (.log + console), set_seed, checkpointing, count_parameters
│   └── models/
│       ├── __init__.py                 # Registry và build_model()
│       ├── base.py                     # BaseAudioClassifier (hỗ trợ cả waveform và spectrogram)
│       ├── ced_tiny.py                 # Single CED-Tiny Audio-Transformer (~5.41M)
│       └── dual_ced_tiny.py            # Dual-CED-Tiny (~10.80M) gồm 2 expert + learned router
├── scripts/
│   ├── __init__.py
│   ├── verify_models.py                # Kiểm tra forward pass, shape và ngân sách tham số
│   ├── run_training.py                 # Runner huấn luyện Dual-CED-Tiny hoặc từng chuyên gia
│   └── evaluate_models.py              # Đánh giá checkpoint theo 6 metrics và bảng phân rã 6 mức SNR
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Fixtures dùng chung cho pytest
│   ├── test_models.py                  # Unit tests forward pass và shape cho Dual-CED-Tiny
│   ├── test_router.py                  # Unit tests cơ chế định tuyến SNR và soft blending
│   ├── test_dataset.py                 # Unit tests dataset loader và RMS normalization
│   ├── test_metrics.py                 # Unit tests acc, precision, recall, f1, mAP
│   └── test_validation_selection.py    # Unit tests chọn checkpoint theo val acc & hòa giữ epoch trước
├── main.py                             # CLI điều khiển trung tâm toàn bộ tác vụ
├── requirements.txt                    # Danh mục thư viện phụ thuộc
└── README.md                           # Tài liệu kỹ thuật chi tiết
```

---

## 2. Thông Số Kiến Trúc & Ngân Sách Tham Số

| Tên mô hình | Vai trò | Số tham số | Trainable | Cấu hình cốt lõi |
| :--- | :--- | :---: | :---: | :--- |
| **`ced_tiny_low`** | Chuyên gia Low-SNR | **`~5.412M`** | `~5.412M` | 12 tầng ($D=192$, $H=3$), GREP + Alpha Post-LN. |
| **`ced_tiny_high`** | Chuyên gia High-SNR | **`~5.412M`** | `~5.412M` | Cùng kiến trúc, được chuyên biệt hóa cho SNR cao. |
| **`dual_ced_tiny`** | **Dual-Expert System** | **`~10.80M`** | `~10.80M` | Hai expert + learned spectral-token router; soft blending. |

### Sơ đồ luồng dữ liệu (Dataflow):

```
                        Âm thanh đầu vào: Waveform [B, 64000] (16kHz, 4.0s)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
                       [6 acoustic cues]             [LogMel Frontend]
                               │                    64 Mels × 398 Frames [B, 1, 64, 398]
                    │                                     │
                    │                    ┌────────────────┴────────────────┐
                    │                    ▼                                 ▼
                    │           [S1: CED-Tiny Low]                [S2: CED-Tiny High]
                    │           (~5.412M parameters)               (~5.412M parameters)
                    │                    │                                 │
                    │                    ▼                                 ▼
                    │             Logits S1 [B, 36]                 Logits S2 [B, 36]
                    │                    │                                 │
                    └────────────────────┼─────────────────────────────────┘
                                         ▼
                             [Learned token gate]
               • α = softmax(MLP([acoustic features, projected CLS]))_High
               • z = (1 - α) · z_low + α · z_high
                                         │
                                         ▼
                            Đầu ra: Logits cuối cùng [B, 36]
```

---

## 3. Siêu Tham Số Huấn Luyện (Áp dụng chuẩn `imp/mix_dataset_v2`)

Hệ thống tuân thủ nghiêm ngặt các tham số từ `imp/mix_dataset_v2`:

* **Số epochs:** Tối đa 15 epoch theo recipe `imp/moe-v2`, early stopping patience 8; có thể chạy thêm 6 epoch Stage-2 cho High expert.
* **Kích thước lô hiệu dụng (Effective Batch Size):** `64` (Micro-batch `32` × Gradient Accumulation `2`).
* **Bộ tối ưu (Optimizer):** `AdamW`, `weight_decay = 0.05`.
* **Learning Rate:** backbone `5e-5`, classifier heads `1.5e-4`, learned router `2e-4`.
* **Lịch điều phối LR (Scheduler):** `Cosine Annealing` với `2 epochs` linear warmup.
* **Định dạng dữ liệu đầu vào:** FRCRN-denoised cache `.npy` (16kHz, mono, 4.0 giây = 64.000 samples), chuẩn hóa RMS về `0.07925` trước khi vào frontend.
* **Loss theo `imp/moe-v2`:** `L = 0.35 L_CE(fused) + 0.65 L_KD + 0.20 L_route`; `L_KD = 0.65 L_KD_fused + 0.35 L_KD_experts`, `T=2`. Hai teacher được pha theo `w=sigmoid((SNR-15)/2.5)` cho loss fused; nhánh Low ghép Fold0, High ghép EMA. Route BCE dùng cùng soft target SNR, chỉ ở train. Stage-2 High expert tiếp tục CE + KD từ EMA.
* **Tăng cường dữ liệu:** Mixup đồng thời waveform, nhãn, SNR và phân phối xác suất teacher (`alpha=0.4`, `prob=0.8`), không clamp lambda về expert trội; SpecAugment (`F=2x8`, `T=2x24`); Label Smoothing `0.1`. BEATs teacher đóng băng, không tham gia val/inference.
* **Khi `--no-kd`:** fused CE + 0.5 regime-weighted expert CE + 0.2 route BCE.
* **Student CED:** GREP điều biến 2D relative-position bias theo query; Alpha Post-LN dùng residual scale $\alpha=(2L)^{1/4}$, như `AlphaBlock` trong `imp/moe-v2`.
* **Độ chính xác số thực:** `bfloat16` Automatic Mixed Precision (AMP).
* **Chọn mô hình tốt nhất (`best_model.pt`):** Dựa trên `validation accuracy` cao nhất; nếu điểm số bằng nhau, **giữ epoch đến trước** ("earliest epoch on ties").
* **Suy luận Overlap:** Đánh giá multi-crop `3 × 3.2 s` tại các mốc `0.0 / 0.4 / 0.8 s`, trung bình logits. BEATs không được nạp ở inference. Checkpoint được chọn theo 1-pass validation accuracy; overlap chỉ dùng cho báo cáo.

---

## 4. Hướng Dẫn Sử Dụng CLI

Chạy từ thư mục gốc `imp/dual-ced-tiny`:

### 4.1. Liệt kê danh mục mô hình & tham số
```powershell
python main.py list-models
```

### 4.2. Kiểm tra xác thực cấu trúc & shape (Forward Verification)
```powershell
python main.py verify
```

### 4.3. Huấn luyện mô hình (Training)
```powershell
# Chạy smoke test 1 epoch nhanh trên dữ liệu synthetic
python main.py train --model dual_ced_tiny --smoke

# Chạy theo config hiện tại: tối đa 15 epochs + Stage-2 High expert
python main.py train --model dual_ced_tiny

# Huấn luyện riêng biệt nhánh Low-Expert hoặc High-Expert
python main.py train --model ced_tiny_low --epochs 15
python main.py train --model ced_tiny_high --epochs 15
```

### 4.4. Đánh giá Checkpoint (Evaluation & SNR Breakdown)
```powershell
# Đánh giá 1-pass + overlap trên tập eval kèm bảng phân rã 6 mức SNR
python main.py eval --model dual_ced_tiny --checkpoint checkpoints_moev2_router/dual_ced_tiny/best_model.pt

# Chỉ đánh giá 1-pass (tắt overlap)
python main.py eval --model dual_ced_tiny --checkpoint checkpoints_moev2_router/dual_ced_tiny/best_model.pt --no-overlap

# Tùy biến overlap: 3 cửa sổ 3.2 s, trung bình logits
python main.py eval --model dual_ced_tiny --overlap --num-windows 3 --crop-seconds 3.2 --aggregate logits
```
Lưu ý: tập `test` chỉ dùng được FRCRN cache nếu có `datasets/mix-frcrn/test.npy`; nếu thiếu, loader tự quay về WAV mixture gốc.

---

## 5. Chạy Kiểm Thử Tự Động (Unit Tests)

Dự án tích hợp đầy đủ bộ unit test với `pytest`:

```powershell
python -m pytest tests -q
```
Các bài test bao gồm:
* `test_models.py`: Kiểm tra shape forward ([2, 64000] -> [2, 36]) và ngân sách tham số (cập nhật theo GREP + Post-LN).
* `test_router.py`: Kiểm tra ước lượng SNR và cơ chế chuyển mạch router.
* `test_dataset.py`: Kiểm tra dataset loader, multi-hot target và chuẩn hóa RMS.
* `test_metrics.py`: Kiểm tra tính toán 6 metrics và bảng phân rã SNR.
* `test_inference.py`: Kiểm tra multi-crop overlap (starts, dedupe, crop ngắn hơn cửa sổ, trung bình logits).
* `test_validation_selection.py`: Kiểm tra quy tắc lưu checkpoint theo val acc và giữ epoch sớm hơn khi hòa.
