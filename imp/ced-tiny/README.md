# CED-Tiny: Huấn luyện & Đánh giá Kiến trúc CED-Tiny trên Hỗn hợp Âm thanh qua FRCRN

Gói mã nguồn tự chứa tại `imp/ced-tiny`, triển khai và thử nghiệm mô hình **CED-Tiny (5.42M tham số) với các cải tiến kiến trúc từ `kd-v5` (Gated Relative Position Bias Attention, Time-Major Spatial-Temporal Re-ordering, 2D Positional Embedding Outer-Addition)**. Trong lần thực nghiệm này, mô hình nhận trực tiếp tín hiệu nhiễu ước lượng $\hat{\mathbf{n}}$ được tách ra từ hỗn hợp âm thanh (Mixture) thông qua bộ lọc **FRCRN (Complex UNet 13.14M tham số, đã huấn luyện trước và đóng băng)** mà không sử dụng chưng cất tri thức (Knowledge Distillation) từ Teacher BEATs.

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
│ Log-Mel Filterbank (64 bins)    │ ──► Mel Spectrogram [B, 1, 64, 398]
│ Per-instance Normalization      │
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ SpecAugment (Freq & Time Mask)  │ ──► [B, 1, 64, 398] (chỉ bật khi train)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ CED-Tiny ViTSBias (12 blocks)   │ ──► Patch Embed 16x16: 4 freq x 24 time patches
│ Time-major Permute + CLS Token  │ ──► Time-First tokens: [B, 97, 192]
│ Gated Relative Position Bias    │ ──► SDPA với T5 bucket relative bias + GRU gate
│ Head phân loại 36 lớp nhiễu    │ ──► Logits [B, 36]
└─────────────────────────────────┘
          │
          ▼
    Soft-Target Cross-Entropy (Label Smoothing = 0.1, Mixup = 0.8)
```

### 1.2. Công thức Toán học

1. **Tách tín hiệu nhiễu ước lượng:**
   $$\hat{\mathbf{s}} = \text{FRCRN}(\mathbf{x}_{\text{mix}}), \quad \hat{\mathbf{n}} = \mathbf{x}_{\text{mix}} - \hat{\mathbf{s}}$$

2. **Chuẩn hóa năng lượng RMS (Energy Normalization):**
   Đưa biên độ clip nhiễu về dải phân phối chuẩn của AudioSet ($\sigma_0 \approx 0.07925$):
   $$\text{RMS}(\hat{\mathbf{n}}) = \sqrt{\frac{1}{T} \sum_{t=1}^T \hat{n}_t^2}$$
   $$\hat{\mathbf{n}}_{\text{norm}} = \hat{\mathbf{n}} \cdot \frac{\sigma_0}{\text{RMS}(\hat{\mathbf{n}}) + \epsilon}$$

3. **Cơ chế Gated Relative Position Bias (kế thừa từ `kd-v5`):**
   - Vị trí tương đối giữa các token $i, j$ được gán vào 320 buckets log-scale với khoảng cách tối đa $K=800$:
     $$B_{ij} = \text{bucket}(i - j, N_{\text{bucket}}=320, d_{\max}=800)$$
     $$\mathbf{P}_{\text{rel}} = \text{Embedding}(B_{ij}) \in \mathbb{R}^{H \times N \times N}$$
   - Cổng điều tiết động GRU-style từ truy vấn $\mathbf{Q}$:
     $$\mathbf{g} = \sigma\left(\text{Linear}(\mathbf{Q})\right)$$
     $$\mathbf{S} = \mathbf{g}_a \odot (\mathbf{g}_b \odot \mathbf{a}_{\text{grep}} - 1.0) + 2.0$$
     $$\text{Bias}_{\text{attn}} = \mathbf{S} \odot \mathbf{P}_{\text{rel}}$$
   - Tích chập chú ý hiệu năng cao (Scaled Dot-Product Attention):
     $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \text{Bias}_{\text{attn}}\right) \mathbf{V}$$

4. **Hàm mất mát phân loại:**
   Sử dụng Soft-Target Cross-Entropy kết hợp làm mịn nhãn (Label Smoothing $\epsilon=0.1$) và Mixup ($\alpha=0.4, p=0.8$):
   $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} \tilde{y}_c \log \frac{\exp(z_c)}{\sum_{j=1}^C \exp(z_j)}$$

---

## 2. Chi tiết Cải tiến Kiến trúc CED-Tiny từ `kd-v5`

| Thành phần | CED Gốc (`mispeech/ced-tiny`) | CED-Tiny Cải tiến (`kd-v5` / Hiện tại) |
|---|---|---|
| **Backbone** | DeiT-Tiny thuần (`patch16_224`) | DeiT-Tiny tùy biến `ViTSBias` |
| **Thứ tự Token** | Freq-major $(F \times T)$ | Time-major $(T \times F)$ sau khi cộng `pos_embed` |
| **Vị trí Tương đối** | Không có (chỉ có absolute separable pos_embed) | Gated Relative Position Bias (T5 buckets + cổng co giãn GRU) |
| **Khởi tạo Bias** | N/A | `bias_init: zero` (khởi đầu thuần CED, học bias thích nghi dần) |
| **Positional Embedding** | Outer-sum 2D `(freq + time)` | Nội suy tuyến tính dải tần $F=4$ và cắt chuỗi thời gian $T=24$ |
| **Residual Blocks** | Pre-LN chuẩn | Pre-LN chuẩn kèm hỗ trợ DeepNorm `AlphaBlock` |
| **Đầu ra Head** | 527 lớp AudioSet | 36 lớp nhiễu môi trường của bài toán MLNC |

---

## 3. Cấu trúc Thư mục

```
imp/ced-tiny/
├── config/
│   └── ced_tiny.yaml        # Cấu hình siêu tham số toàn diện
├── ced_tiny/
│   ├── __init__.py          # Export public API
│   ├── cli.py               # CLI parser: params, smoke, train, evaluate, export
│   ├── config.py            # Dataclass cấu hình kế thừa BaseConfig
│   ├── data.py              # Dataloader Mixture-36l, sampler cân bằng, bộ lọc Unseen
│   ├── evaluate.py          # Đo đạc 9 chỉ số tổng hợp, 6 mức SNR, bộ lọc Unseen
│   ├── frontend.py          # LogMel 64-bin chuẩn hóa per-instance & SpecAugment
│   ├── loss.py              # SoftTargetCrossEntropy hỗ trợ Label Smoothing & Mixup
│   ├── model.py             # CEDTiny & FRCRNCEDPipeline (FRCRN + CED-Tiny)
│   ├── pipeline.py          # Quy trình huấn luyện, phục hồi best checkpoint & xuất CSV
│   ├── runtime.py           # Quản lý thiết bị & autocast context (bf16/fp16)
│   ├── trainer.py           # CEDTinyTrainer (kế thừa Trainer, bit-for-bit resume)
│   └── vit_bias.py          # ViTSBias, BiasAttention, AlphaBlock & load_ced_tiny
├── tests/
│   ├── conftest.py          # Cấu hình đường dẫn pytest
│   ├── test_config.py       # Kiểm thử cấu hình & validation rules
│   ├── test_data.py         # Kiểm thử LabelBalancedSampler & Unseen filter
│   ├── test_model.py        # Kiểm thử forward, backward, predict_proba, RMS norm
│   └── test_trainer.py      # Kiểm thử chu kỳ fit 2 epoch & resume trạng thái bit-for-bit
├── run.py                   # Điểm vào dòng lệnh (CLI)
└── README.md                # Tài liệu kỹ thuật này
```

---

## 4. Bảng Tham số Mô hình Chi tiết

Kiểm tra bằng lệnh `python run.py params`:

```
=================================================================
CED-Tiny Pipeline Configuration & Parameters Summary
=================================================================
FRCRN config:        config/frcrn.yaml
FRCRN Checkpoint:    pretrained/FRCRN/best_e037_20261020T203958Z.pt
Target RMS:          0.07925
CED Architecture:    deit_tiny_patch16_224 (init=ced-tiny)
CED Positional Bias: zero (T5 buckets + GRU gate)
Frontend:            LogMel (64 mels, 398 frames)
Learning Rate:       0.0001
Batch Size:          32 (acc=1)
Epochs:              15
Selection Metric:    macro_acc
-----------------------------------------------------------------
CED-Tiny Total Parameters:     5.42M
CED-Tiny Trainable Parameters: 5.42M
  - patch embed (conv 16x16)           :     49,344 (0.049M)
  - CLS + absolute position embedding  :     18,816 (0.019M)
  - attention qkv + proj               :  1,778,688 (1.779M)
  - bucket + position bias embedding   :        960 (0.001M)
  - GRU gate (gated bias)              :      6,276 (0.006M)
  - MLP (fc1 + fc2)                    :  3,550,464 (3.550M)
  - LayerNorm                          :      9,600 (0.010M)
  - classifier head                    :      6,948 (0.007M)
-----------------------------------------------------------------
FRCRN Parameters (Frozen):     13.14M
=================================================================
```

---

## 5. Hướng dẫn Sử dụng Dòng Lệnh (CLI)

```powershell
# 1. Kiểm tra tham số & cấu hình hệ thống
python imp/ced-tiny/run.py params

# 2. Chạy smoke test nhanh (1 epoch, tập mẫu nhỏ tổng hợp)
python imp/ced-tiny/run.py smoke

# 3. Huấn luyện đầy đủ trên tập dữ liệu hỗn hợp qua FRCRN
python imp/ced-tiny/run.py train --config imp/ced-tiny/config/ced_tiny.yaml

# 4. Huấn luyện với thiết bị chỉ định hoặc chỉ định run-id để resume
python imp/ced-tiny/run.py train --device cuda:0 --run-id ced-tiny-2026-10-29-17-00-00

# 5. Đánh giá checkpoint đã huấn luyện trên tập validation hoặc test
python imp/ced-tiny/run.py evaluate --checkpoint output/checkpoint/<run-id>/ced_tiny/last_best_checkpoint.pt --split validation

# 6. Xuất checkpoint CED-Tiny độc lập để triển khai suy luận
python imp/ced-tiny/run.py export --checkpoint output/checkpoint/<run-id>/ced_tiny/last_best_checkpoint.pt --output pretrained/ced_tiny_ft.pt
```

---

## 6. Cấu trúc Tệp Kết quả Xuất ra

Đầu ra của quy trình được tổ chức chặt chẽ theo quy ước `RunPaths`:

```
output/
├── log/
│   └── <run-id>.log
├── checkpoint/
│   └── <run-id>/
│       ├── ced_tiny/
│       │   └── last_best_checkpoint.pt   # Checkpoint có macro_acc validation cao nhất
│       └── resume/
│           └── ced_tiny.pt               # Trạng thái đầy đủ của epoch gần nhất (optimizer, scheduler, epoch)
└── result/
    └── <run-id>/
        ├── result.json                   # Tổng hợp toàn bộ metrics, history và unseen
        ├── result.csv                    # Kết quả tổng thể validation & unseen
        ├── snr_breakdown_val.csv         # Chi tiết 9 metrics cho 6 mức SNR (-5, 0, 5, 10, 15, 20 dB)
        └── snr_breakdown_unseen.csv      # Chi tiết theo SNR cho tập Unseen
```


---

## 7. So Sánh Hiệu Năng: CED-Tiny Cải Tiến vs. Các Pipeline BEATs vs. CED-Tiny Nguyên Bản

### 7.1. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Benchmark Table)

Dưới đây là so sánh toàn diện giữa **CED-Tiny cải tiến** (`kd-v5` / hiện tại), **các biến thể BEATs** và **CED-Tiny nguyên bản** dựa trên dữ liệu thực nghiệm lưu tại `.log/experiments.md` và `imp/kd-v4`:

| Mô hình / Pipeline | Số lượng tham số | Checkpoint size | Val Macro-Acc | Val mAP | Chênh lệch Train - Val | Đánh giá & Khả năng triển khai |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Teacher BEATs 12L** (`best.pt`) | ~90M | ~360 MB | **0.694** | **0.766** | 0.28 - 0.35 | **Trần hiệu năng cao nhất**, nhưng mô hình quá nặng, tốn GPU (>8GB VRAM) |
| **BEATs cắt 6 lớp (BEATs-6L)** (`run_beats_trunc`, E3) | 47.8M | ~191 MB | 0.577 | 0.630 | 0.368 (Train 0.945) | Bị overfit nghiêm trọng trên tập 36 lớp; tham số vẫn còn lớn |
| **Fine-tune BEATs Đợt 4** (`imp/ft-beat`) | ~90M | ~360 MB | 0.482 | 0.494 | ~0.420 | Bị overfit khi fine-tune trực tiếp trên hỗn hợp; Top-1 Acc: 49.74% |
| **CED-Tiny Nguyên bản** (DeiT-Ti AudioMAE) | 5.42M | ~21.7 MB | 0.463 | 0.501 | 0.351 - 0.500 | Dễ sụp đổ gradient nếu bật Dropout ($std=12.5$); thiếu context thời gian |
| **CED-Tiny Cải tiến (`kd-v5` / Hiện tại)** | **5.42M** | **~21.7 MB** | **0.629** | **0.701** | **0.223** (Thấp nhất) | 🏆 **Tối ưu toàn diện**: Nhỏ hơn BEATs 16.6x, mAP đạt 70.1%, chống overfit tốt nhất |

### 7.2. Phân Tích Cơ Chế Vì Sao CED-Tiny Cải Tiến Vượt Trội Hơn

#### 1. So với các Pipeline họ BEATs (90M / 47.8M)
* **Khắc phục triệt để Overfitting**: Tập dữ liệu `mixture-36l` có khoảng 30,240 mẫu huấn luyện. Các mô hình BEATs (47.8M đến 90M tham số) quá lớn so với kích thước bài toán, dẫn đến việc mô hình "học vẹt" dữ liệu train (Train Acc đạt 94% - 99%), làm chênh lệch giữa Train và Val lên tới hơn 0.40 - 0.46. Ngược lại, CED-Tiny chỉ có **5.42M tham số**, kết hợp với Mixup ($0.8$) và Label Smoothing ($0.1$) đạt được điểm cân bằng bias-variance tối ưu (khoảng cách Train - Val chỉ **0.223**).
* **Hiệu suất tham số (Parameter Efficiency)**: Với chỉ bằng **6% dung lượng tham số** của Teacher BEATs 12L, CED-Tiny đạt mAP **70.1%** (so với 76.6% của BEATs 90M và vượt xa mức 63.0% của BEATs-6L 47.8M).
* **Khả năng triển khai thực tế**: Checkpoint CED-Tiny chỉ **~21.7 MB** (so với >360 MB của BEATs), tốc độ suy luận đạt trên **6,000 mẫu/giây**, cho phép tích hợp trực tiếp vào các thiết bị nhúng (Edge AI) hoặc dịch vụ thời gian thực có độ trễ cực thấp.

#### 2. So với CED-Tiny nguyên bản (Vanilla AudioMAE / DeiT-Tiny)
* **Time-Major Spatial-Temporal Reordering**: Ở bản gốc, các patch mel-spectrogram được xếp theo thứ tự tần số $(F \times T)$. Bản cải tiến chuyển vị `transpose(1, 2)` sang thứ tự thời gian $(T \times F)$, giúp các token tự chú ý (self-attention) xử lý tín hiệu âm thanh theo đúng dòng thời gian thực.
* **Gated Bucketed Relative Position Bias**: Bản gốc chỉ có positional embedding tĩnh 2D. Bản cải tiến tích hợp T5 bucket relative position bias (320 buckets, 800 khoảng cách) được chia sẻ 12 tầng và điều biến thích ứng bằng cổng GRU ($Q \cdot W_g$), giúp mô hình nắm bắt độ dài và khoảng cách tương đối của sự kiện âm thanh.
* **Loại bỏ Dropout & Ổn định Kích hoạt**: Phân tích chẩn đoán tại `imp/kd-v4` cho thấy kích hoạt đầu vào của CED có độ lệch chuẩn rất lớn ($std \approx 12.5$ so với $0.55$ của DeiT ImageNet). Việc dùng Dropout ở bản gốc gây ra hiện tượng sụp đổ gradient (Val Acc chỉ đạt 2.8% - 3.8%). Bản cải tiến loại bỏ hoàn toàn Dropout, sử dụng chuẩn hóa Z-score theo từng mẫu (Instance Norm) kết hợp Mixup Beta(0.4, 0.4), giúp quá trình huấn luyện hoàn toàn ổn định và hội tụ nhanh chóng.

### 7.3. Kết Luận: Hiệu Năng Cái Nào Tốt Hơn?
* **Về Độ chính xác thuần túy (Raw Accuracy / mAP)**:
  - Teacher BEATs 12L (90M) giữ trần lý thuyết cao nhất ($76.6\%$ mAP) nhưng không thực tế để triển khai thực tế do quá cồng kềnh.
  - **CED-Tiny cải tiến đạt mAP 70.1%**, vượt qua toàn bộ các biến thể rút gọn của BEATs (BEATs 6L đạt 63.0%, BEATs 4L chỉ đạt 46.2%) và vượt trội hoàn toàn so với CED-Tiny nguyên bản (~50.1%).
* **Về Tỷ số Hiệu năng / Chi phí tính toán (Pareto Efficiency)**:
  - **CED-Tiny Cải tiến là lựa chọn tốt nhất toàn diện**: Giảm **16.6 lần kích thước**, chống overfitting tốt nhất, suy luận tức thì, và duy trì hiệu năng tiệm cận mô hình 90M.
