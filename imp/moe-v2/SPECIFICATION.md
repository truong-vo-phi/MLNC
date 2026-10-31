# Bản Đặc Tả Kỹ Thuật (Technical Specification)
## Hệ Thống `Champion Pipeline v2: SplitFFN + Teachers (ZERO ORACLE)`

**Mã định danh cấu hình:** `Champion Pipeline v2: CED-SingleMoE-SplitFFN (BEATs Dual-Teacher KD) + Teachers (Self-Contained Router)`  
**Thư mục mã nguồn:** `D:\PROJECT\MLNC\imp\moe-v2`  
**Các tệp thực thi chính:**
- Mô hình Student 1-Backbone: [`src/models/moe_split.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/src/models/moe_split.py) (`CEDSingleMoESplitFFN`, `SplitMoEFFN`)
- Bộ định tuyến nội tại: [`src/router.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/src/router.py) (`SelfContainedSpectralTokenRouter`)
- Trích xuất đặc trưng & Frontend: [`src/frontend.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/src/frontend.py) (`LogMel`, `AcousticFeatureExtractor`, `SpecAugment`)
- Khung xương sống ViT 2D Bias: [`src/models/vit_bias.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/src/models/vit_bias.py) (`ViTSBias`, `BiasAttention`)
- Huấn luyện Chưng cất Đa thầy: [`src/kd_trainer.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/src/kd_trainer.py) (`KDTrainer`)
- Điều phối Suy luận Pipeline: [`src/pipeline.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/src/pipeline.py) (`ChampionPipelineV2`) & [`champion_pipeline.py`](file:///d:/PROJECT/MLNC/imp/moe-v2/champion_pipeline.py)

---

## 1. Tổng Quan & Mục Tiêu Thiết Kế

**`Champion Pipeline v2: SplitFFN + Teachers (ZERO ORACLE)`** là kiến trúc phân loại âm thanh môi trường kháng nhiễu (36 lớp, đa mức SNR từ $-5\text{ dB}$ đến $20\text{ dB}$) được thiết kế nhằm giải quyết triệt để hai điểm nghẽn của hệ thống `imp/moe` thế hệ đầu:
1. **Hợp nhất 2 mô hình Student rời (`10.84M` tham số) về duy nhất 1 Backbone (`6.62M` tham số):** Chia sẻ 100% 10 khối Transformer đầu tiên (`Blocks 0..9`) và toàn bộ cơ chế `BiasAttention` 12 tầng; chỉ phân tách nhánh chuyên gia FFN tại 2 khối sâu nhất (`Blocks 10 & 11`) cùng `Dual Classifier Head`.
2. **Loại bỏ hoàn toàn phụ thuộc vào nhãn Oracle SNR (`Zero-Oracle`):** Tự động ước lượng trọng số chế độ nhiễu liên tục $\alpha \in (0, 1)$ trực tiếp từ hình thái phổ - thời gian của tín hiệu nhiễu ước lượng $\hat{\mathbf{n}}$ và token ẩn tại Block 9, không cần truyền `snr_db` thật hay tín hiệu hỗn hợp gốc $\mathbf{x}_{\text{mix}}$ lúc suy luận.

### Bảng Thông Số Hệ Thống Tổng Quát

| Hạng mục | Thông số kỹ thuật | Ghi chú |
| :--- | :--- | :--- |
| **Đầu vào tín hiệu** | $\hat{\mathbf{n}} = \mathbf{x}_{\text{mix}} - \hat{\mathbf{s}} \in \mathbb{R}^{B \times 64000}$ | Nhiễu tách qua FRCRN đóng băng ($16\text{ kHz}, 4.0\text{ s}$) |
| **Student Backbone** | `CEDSingleMoESplitFFN` (`deit_tiny_patch16_224` + 2D GREP Bias) | $12\text{ layers}, D=192, H=3, d_{\text{ffn}}=768$ |
| **Cấu trúc MoE trong Student** | `SplitMoEFFN` tại **Blocks 10 & 11** + `Dual Head` | `1 Shared FFN` + `2 Routed FFNs (Low/High)` |
| **Số tham số Student (Trainable)** | **`6.62M`** (`6,617,322` tham số) | Giảm **39.0%** so với 2 Student rời (`10.84M`) |
| **Giáo viên 1 (`Teacher Fold0`)** | `BEATClassifier` 12L (`beats_teacher_fold0_weights.pt`) | `90.34M` tham số (Đóng băng — Chuyên gia Low/Mid-SNR) |
| **Giáo viên 2 (`Teacher EMA`)** | `BEATClassifier` 12L (`beats_teacher_ema_standalone.pt`) | `90.34M` tham số (Đóng băng — Chuyên gia High-SNR) |
| **Tổng tham số toàn Pipeline** | **`187.30M`** (`6.62M` Trainable + `180.68M` Frozen) | Hỗ trợ chạy cả `self_contained` lẫn `standalone` (`6.62M`) |

---

## 2. Sơ Đồ Kiến Trúc Luồng Dữ Liệu (End-to-End Architecture)

```
                         Waveform nhiễu ước lượng n̂ [B, 64000] (16 kHz, 4.0s)
                                                 │
                                  Chuẩn hóa RMS (σ_0 = 0.07925)
                                                 │
                      ┌──────────────────────────┴──────────────────────────┐
                      ▼                                                     ▼
        3-Way Temporal TTA (±50 ms)                               Lượt chạy gốc w_0 [B, 64000]
      w_0 (0ms), w_+50 (+800), w_-50 (-800)                                 │
                      │                                     ┌───────────────┴───────────────┐
        ┌─────────────┼─────────────┐                       ▼                               ▼
        ▼             ▼             ▼         AcousticFeatureExtractor             LogMel Frontend
  Teacher Fold0  Teacher EMA   CED-SplitFFN    6 đặc trưng hình thái phổ         [B, 1, 64, 398] (float32)
   (90.34M)       (90.34M)       (6.62M)        f_spec [B, 6]                               │
        │             │             │                       │                               ▼
        │             │             │                       │                    PatchEmbed (16×16) + 2D Pos
        │             │             │                       │                      Tokens X_0 [B, 97, 192]
        │             │             │                       │                               │
        │             │             │                       │                               ▼
        │             │             │                       │                      Shared Blocks 0 -> 9
        │             │             │                       │                    (100% Dùng chung, 2D GREP)
        │             │             │                       │                               │
        │             │             │                       │                    ┌──────────┴──────────┐
        │             │             │                       │                    ▼                     ▼
        │             │             │                       │             CLS Token h_9            Tokens X_10
        │             │             │                       │               [B, 192]              [B, 97, 192]
        │             │             │                       │                    │                     │
        │             │             │                       └─────────┬──────────┘                     │
        │             │             │                                 ▼                                │
        │             │             │                  SelfContainedSpectralTokenRouter                │
        │             │             │                     α = Softmax(MLP)_1 ∈ (0, 1)                  │
        │             │             │                                 │                                │
        │             │             │                                 ├────────────────────────────────┤
        │             │             │                                 │                                ▼
        │             │             │                                 │                  SplitMoEFFN Blocks 10 & 11
        │             │             │                                 │               0.5·FFN_shared + 0.5·[(1-α)·FFN_low
        │             │             │                                 │                                   + α·FFN_high]
        │             │             │                                 │                                │
        │             │             │                                 │                                ▼
        │             │             │                                 │                       Dual Classifier Head
        │             │             │                                 │                  z_s = (1-α)·Head_low + α·Head_high
        │             │             ◄─────────────────────────────────┼────────────────────────────────┘
        ▼             ▼             ▼                                 │
   z̄_T0 [B,36]   z̄_Tema [B,36]  z̄_S [B,36]                            │
        │             │             │                                 │
        ▼             ▼             ▼                                 │
   Softmax/1.20  Softmax/1.20  Softmax/1.40                           │
     P_T0          P_Tema        P_Student                            │
        │             │             │                                 │
        └─────────────┼─────────────┴─────────────────────────────────┘
                      ▼
       Zero-Oracle Dynamic Soft Blending:
         w_S     = 0.25
         w_T0(α) = (1 - α)·0.45 + α·0.30
         w_Em(α) = (1 - α)·0.30 + α·0.45
         P_final = w_S·P_Student + w_T0(α)·P_T0 + w_Em(α)·P_Tema
                      │
                      ▼
            Nhãn dự đoán: ŷ = argmax(P_final)
```

---

## 3. Đặc Tả Toán Học & Cấu Trúc Từng Thành Phần

### 3.1. Chuẩn Hóa RMS & Frontend Phổ Log-Mel (`src/frontend.py`)
1. **Chuẩn hóa năng lượng RMS (`rms_normalize`):**
   Đưa biên độ hiệu dụng của mọi mẫu âm thanh $\hat{\mathbf{n}} \in \mathbb{R}^{B \times 64000}$ về hằng số chuẩn $\sigma_0 = 0.07925$:
   $$\hat{\mathbf{n}}_{\text{norm}} = \hat{\mathbf{n}} \cdot \frac{\sigma_0}{\sqrt{\frac{1}{T}\sum_{t=1}^T \hat{n}_t^2} + 10^{-8}}$$
2. **Trích xuất phổ Log-Mel (`LogMel`):**
   - Biến đổi STFT với $N_{\text{fft}} = 512$, độ dài cửa sổ Hann $L_{\text{win}} = 400$ ($25\text{ ms}$), bước nhảy $L_{\text{hop}} = 160$ ($10\text{ ms}$), `center=False`.
   - Nhân ma trận bộ lọc Mel chuẩn HTK ($F = 64\text{ bins}$, dải tần $20\text{ Hz} - 8000\text{ Hz}$) và lấy logarit tự nhiên:
     $$\mathbf{M}_{\text{raw}} = \ln\!\left(\mathbf{W}_{\text{mel}} |\text{STFT}(\hat{\mathbf{n}}_{\text{norm}})|^2 + 10^{-6}\right) \in \mathbb{R}^{B \times 64 \times T_f}$$
   - Chuẩn hóa Z-score độc lập trên từng mẫu (Instance Normalization) và cắt cửa sổ $398$ khung thời gian:
     $$\mathbf{M} = \text{Crop}_{398}\!\left(\frac{\mathbf{M}_{\text{raw}} - \mu_{\text{instance}}}{\sigma_{\text{instance}} + 10^{-5}}\right) \in \mathbb{R}^{B \times 1 \times 64 \times 398}$$
   - *Quy tắc độ chính xác số học:* Khối `LogMel` luôn được ép chạy ở `float32` (`torch.autocast(enabled=False)`) để tránh tràn số khi tính phương sai và logarit.

---

### 3.2. Trích Xuất 6 Đặc Trưng Hình Thái Âm Học Nội Tại (`AcousticFeatureExtractor`)
Để nhận biết mức độ suy biến SNR và giả ảnh rò rỉ tiếng nói (speech leakage) từ FRCRN mà **không cần nhãn `snr_db` hay tín hiệu hỗn hợp $\mathbf{x}_{\text{mix}}$**, mô-đun tính phổ công suất STFT $P(f, t) = |\text{STFT}(\hat{\mathbf{n}}_{\text{norm}})|^2 + \epsilon \in \mathbb{R}^{B \times 257 \times 401}$ và trích xuất 6 đặc trưng vật lý $\mathbf{f}_{\text{spec}} = [f_1, f_2, f_3, f_4, f_5, f_6] \in \mathbb{R}^{B \times 6}$:

1. **Log-RMS ($f_1$):** $f_1 = \ln(\text{RMS}(\hat{\mathbf{n}}_{\text{norm}}) + \epsilon)$
2. **Crest Factor ($f_2$):** Tỷ lệ biên độ đỉnh trên giá trị hiệu dụng:
   $$f_2 = \frac{\max_{t} |\hat{n}_{\text{norm}, t}|}{\text{RMS}(\hat{\mathbf{n}}_{\text{norm}}) + \epsilon}$$
3. **Frame-wise Energy Kurtosis ($f_3$):** Với năng lượng từng khung thời gian $E_t = \sum_{f=0}^{256} P(f, t)$:
   $$f_3 = \frac{1}{T_f}\sum_{t=1}^{T_f} \left(\frac{E_t - \mu_E}{\sigma_E + \epsilon}\right)^4$$
   *(Ở vùng SNR cao $15\text{–}20\text{ dB}$, tiếng nói gốc mạnh gấp $31.6\times - 100\times$ nhiễu nền; phần âm tiết tiếng nói rò rỉ qua mặt nạ FRCRN tạo ra các đỉnh xung năng lượng cục bộ có Kurtosis rất lớn so với nhiễu nền dừng ở $-5\text{..}5\text{ dB}$).*
4. **Spectral Flatness / Wiener Entropy ($f_4$):** Đo tính chất phẳng của phổ (nhiễu trắng/nền rộng có độ phẳng cao, cấu trúc hài âm có độ phẳng thấp):
   $$f_4 = \frac{1}{T_f}\sum_{t=1}^{T_f} \frac{\exp\!\left(\frac{1}{257}\sum_{f=0}^{256} \ln P(f, t)\right)}{\frac{1}{257}\sum_{f=0}^{256} P(f, t)}$$
5. **Low-to-Speech Sub-band Ratio ($f_5$):** Tỷ số log năng lượng dải tần thấp (`bins 0..9`, $<300\text{ Hz}$) trên dải tần thoại (`bins 10..108`, $300\text{–}3400\text{ Hz}$):
   $$f_5 = \ln\frac{\sum_{t}\sum_{f=0}^{9} P(f, t) + \epsilon}{\sum_{t}\sum_{f=10}^{108} P(f, t) + \epsilon}$$
6. **High-to-Speech Sub-band Ratio ($f_6$):** Tỷ số log năng lượng dải tần cao (`bins 109..256`, $>3400\text{ Hz}$) trên dải tần thoại (`bins 10..108`):
   $$f_6 = \ln\frac{\sum_{t}\sum_{f=109}^{256} P(f, t) + \epsilon}{\sum_{t}\sum_{f=10}^{108} P(f, t) + \epsilon}$$

---

### 3.3. Khung Xương Sống Dùng Chung (`Blocks 0..9`) & `BiasAttention` (`src/models/vit_bias.py`)
* **Tham chiếu học thuật:** *CED (Dinkel et al., ICASSP 2024, arXiv:2308.11957)* & *BEATs (Chen et al., ICML 2023, arXiv:2212.09058)*.
* **Patch Embedding & Lưới Vị trí 2D:**
  - Lớp `Conv2d(1, 192, kernel_size=16, stride=16)` ánh xạ $\mathbf{M} \in \mathbb{R}^{B \times 1 \times 64 \times 398}$ thành $4\text{ (tần số)} \times 24\text{ (thời gian)} = 96$ patch tokens.
  - Trọng số vị trí `pos_embed` được tái tạo từ tổng ngoài (outer sum) giữa `freq_pos_embed` (nội suy tuyến tính về $4$ patches) và `time_pos_embed` (cắt $24$ patches đầu) của `mispeech/ced-tiny`.
  - Ghép thêm token phân loại `cls_token` ở đầu chuỗi $\implies \mathbf{X}_0 \in \mathbb{R}^{B \times 97 \times 192}$.
* **Cơ chế `BiasAttention` (Gated Relative Position Bias — GREP):**
  - Khoảng cách vị trí tương đối $j - i$ giữa các token được lượng tử hóa logarit vào $N_{\text{bucket}} = 320$ buckets (khoảng cách tối đa $D_{\max} = 800$).
  - Chỉ **Block 0** sở hữu bảng nhúng `relative_attention_bias` (`nn.Embedding(320, 3)`) và lưu vào vùng nhớ chia sẻ `SharedBias` ($\mathbf{P}_{\text{rel}} \in \mathbb{R}^{1 \times 3 \times 97 \times 97}$) để dùng chung cho toàn bộ 12 khối.
  - Tại mỗi khối $l \in \{0..11\}$, vector truy vấn $\mathbf{Q} \in \mathbb{R}^{B \times 3 \times 97 \times 64}$ điều biến biên độ bias qua cổng phi tuyến GREP ($d_{\text{gate}} = 8$):
    $$\mathbf{g} = \sigma\!\left(\text{Linear}_{64 \to 8}(\mathbf{Q})\right) \in \mathbb{R}^{B \times 3 \times 97 \times 2 \times 4}, \quad [\mathbf{g}_a, \mathbf{g}_b] = \text{chunk}\!\left(\sum_{k=1}^4 \mathbf{g}_{..., k}, 2\right)$$
    $$\mathbf{S} = \mathbf{g}_a \odot (\mathbf{g}_b \odot \mathbf{a}_{\text{grep}} - 1.0) + 2.0, \quad \mathbf{B}_{\text{attn}} = \mathbf{S} \odot \mathbf{P}_{\text{rel}}$$
    $$\text{BiasAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Proj}\!\left(\text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{64}} + \mathbf{B}_{\text{attn}}\right)\mathbf{V}\right)$$
* **10 Khối Dùng Chung (`Blocks 0..9`):**
  Toàn bộ 10 khối Transformer đầu tiên (`Blocks 0..9`) được giữ nguyên cấu trúc Pre-LayerNorm và chia sẻ 100% trọng số cho mọi mức SNR:
  $$\mathbf{X}_{l+1} = \mathbf{X}_l + \text{BiasAttn}_l(\text{LN}_1(\mathbf{X}_l)), \quad \mathbf{X}_{l+1} = \mathbf{X}_{l+1} + \text{MLP}_l(\text{LN}_2(\mathbf{X}_{l+1})), \quad l = 0, \dots, 9$$

---

### 3.4. Bộ Định Tuyến Hình Thái Phổ - Token Nội Tại (`SelfContainedSpectralTokenRouter` — `src/router.py`)
* **Tham chiếu học thuật:** *Soft MoE (Puigcerver et al., ICLR 2024, arXiv:2308.00951)*.
* Ngay sau khi hoàn tất **Block 9**, mô hình trích xuất `CLS token` trung gian $\mathbf{h}_9 = \mathbf{X}_{10}[:, 0, :] \in \mathbb{R}^{B \times 192}$ (mang ngữ cảnh âm học bậc cao sau 10 tầng tự chú ý) và kết hợp với 6 đặc trưng vật lý $\mathbf{f}_{\text{spec}} \in \mathbb{R}^{B \times 6}$:
  $$\tilde{\mathbf{h}}_9 = \text{GELU}\!\left(\text{Linear}_{192 \to 16}(\text{LN}(\mathbf{h}_9))\right) \in \mathbb{R}^{B \times 16}$$
  $$\mathbf{u}_{\text{route}} = [\mathbf{f}_{\text{spec}} \,\|\, \tilde{\mathbf{h}}_9] \in \mathbb{R}^{B \times 22}$$
  $$\mathbf{r} = \text{Linear}_{32 \to 2}\!\left(\text{GELU}(\text{LN}(\text{Linear}_{22 \to 32}(\mathbf{u}_{\text{route}})))\right) = [r_{\text{low}}, r_{\text{high}}] \in \mathbb{R}^{B \times 2}$$
  $$\alpha = \frac{\exp(r_{\text{high}})}{\exp(r_{\text{low}}) + \exp(r_{\text{high}})} \in (0, 1)^{B \times 1}$$
* **Hành vi thực nghiệm đã kiểm chứng trên 3,064 mẫu Validation:**
  Giá trị trung bình $\bar{\alpha}$ tự động tăng đơn điệu hoàn hảo theo mức SNR thực tế mà không cần bất kỳ thông tin Oracle nào:
  $$\text{SNR: } -5\text{ dB } (\bar{\alpha}=0.131) \to 0\text{ dB } (0.154) \to 5\text{ dB } (0.199) \to 10\text{ dB } (0.266) \to 15\text{ dB } (0.343) \to 20\text{ dB } (0.436)$$

---

### 3.5. Tầng Chuyên Gia `SplitMoEFFN` tại `Blocks 10 & 11` và `Dual Head` (`src/models/moe_split.py`)
* **Tham chiếu học thuật:**
  - *V-MoE (Riquelme et al., NeurIPS 2021, arXiv:2106.05974 — Mục 3.2: Đặt MoE ở 2 khối cuối cùng "Last-2")*.
  - *DeepSeekMoE (Dai et al., ACL 2024, arXiv:2401.06066 — Tách Shared Expert luôn hoạt động và Routed Experts chuyên biệt)*.
* **Phương trình khối `SplitMoEFFN` ($l \in \{10, 11\}$):**
  Tại Block 10 và Block 11, sau khi đi qua tầng chú ý dùng chung $\mathbf{Y}_l = \mathbf{X}_l + \text{BiasAttn}_l(\text{LN}_1(\mathbf{X}_l))$ và chuẩn hóa $\mathbf{U}_l = \text{LN}_2(\mathbf{Y}_l)$, tín hiệu đi vào 3 mạng FFN song song (`Linear(192, 768) -> GELU -> Linear(768, 192)`):
  - $\text{FFN}_{\text{shared}}^{(l)}$: Chuyên gia dùng chung (luôn nhận trọng số cố định $0.5$), bảo đảm biểu diễn cốt lõi của 36 lớp âm thanh được duy trì xuyên suốt mọi mức SNR.
  - $\text{FFN}_{\text{low}}^{(l)}$: Chuyên gia định tuyến cho miền nhiễu môi trường mạnh ($-5\text{ dB} \to 10\text{ dB}$), nhận trọng số $0.5(1 - \alpha)$.
  - $\text{FFN}_{\text{high}}^{(l)}$: Chuyên gia định tuyến cho miền giả ảnh tách nguồn và rò rỉ tiếng nói ($15\text{ dB} \to 20\text{ dB}$), nhận trọng số $0.5\alpha$.
  $$\text{SplitMoEFFN}_l(\mathbf{U}_l, \alpha) = 0.5 \cdot \text{FFN}_{\text{shared}}^{(l)}(\mathbf{U}_l) + 0.5 \cdot \Big[(1 - \alpha)\,\text{FFN}_{\text{low}}^{(l)}(\mathbf{U}_l) + \alpha\,\text{FFN}_{\text{high}}^{(l)}(\mathbf{U}_l)\Big]$$
  $$\mathbf{X}_{l+1} = \mathbf{Y}_l + \text{SplitMoEFFN}_l(\mathbf{U}_l, \alpha)$$
* **Tính chất bảo toàn kích hoạt khi khởi tạo (Exact Identity Warm-Start):**
  Cả 3 nhánh $\text{FFN}_{\text{shared}}^{(l)}, \text{FFN}_{\text{low}}^{(l)}, \text{FFN}_{\text{high}}^{(l)}$ đều được khởi tạo bằng bản sao trọng số MLP đã tiền huấn luyện của `CED-Tiny` ($\text{MLP}_{\text{CED}}^{(l)}$). Do đó, ở bước khởi đầu ($\forall \alpha \in [0, 1]$):
  $$0.5\,\text{MLP}_{\text{CED}}^{(l)}(\mathbf{U}_l) + 0.5\big[(1-\alpha + \alpha)\,\text{MLP}_{\text{CED}}^{(l)}(\mathbf{U}_l)\big] \equiv 1.0 \cdot \text{MLP}_{\text{CED}}^{(l)}(\mathbf{U}_l)$$
  Nhờ tính chất bất biến này, phương sai kích hoạt đặc trưng của CED-Tiny ($\text{std} = 12.5$) được bảo toàn tuyệt đối ở epoch đầu tiên.
* **Đầu ra `Dual Classifier Head`:**
  Sau Block 11, vector phân loại $\mathbf{h}_{\text{final}} = \text{LN}(\mathbf{X}_{12})[:, 0, :] \in \mathbb{R}^{B \times 192}$ được đưa qua hai đầu tuyến tính $\text{Head}_{\text{low}}, \text{Head}_{\text{high}} \in \mathbb{R}^{192 \times 36}$:
  $$\mathbf{z}_{\text{low}} = \text{Head}_{\text{low}}(\mathbf{h}_{\text{final}}), \quad \mathbf{z}_{\text{high}} = \text{Head}_{\text{high}}(\mathbf{h}_{\text{final}})$$
  $$\mathbf{z}_{\text{student}} = (1 - \alpha)\,\mathbf{z}_{\text{low}} + \alpha\,\mathbf{z}_{\text{high}} \in \mathbb{R}^{B \times 36}$$

### Bảng Thống Kê Chi Tiết Tham Số Của `CEDSingleMoESplitFFN` (`6.62M`)

| Nhóm thành phần | Cấu hình Tensor | Số lượng tham số | Tỷ trọng |
| :--- | :--- | :---: | :---: |
| **Patch Embed + CLS + 2D Pos Embed** | `Conv2d(1, 192, 16, 16)` + `[1, 97, 192]` | `68,160` | `1.03%` |
| **12× BiasAttention + GREP Gate + Bucket Table** | `12 × (QKV + Proj + Gate)` + `Embedding(320, 3)` | `1,785,924` | `26.99%` |
| **10× Shared MLP (`Blocks 0..9`)** | `10 × (192×768 + 768 + 768×192 + 192)` | `2,958,720` | `44.71%` |
| **2× `SplitMoEFFN` (`Blocks 10..11`: Shared + Low + High)** | `2 blocks × 3 experts × 295,872` | `1,775,232` | `26.83%` |
| **Self-Contained Spectral-Token Router** | `LN(192) -> Linear(192,16) -> MLP(22->32->2)` | `4,370` | `0.07%` |
| **LayerNorms + Dual Classifier Heads (`Low` & `High`)** | `LayerNorms` + `2 × Linear(192, 36)` | `24,916` | `0.38%` |
| **Tổng cộng (`CEDSingleMoESplitFFN`)** | **1 Backbone duy nhất** | **`6,617,322` (`6.62M`)** | **`100.0%`** |

---

## 4. Đặc Tả Quy Trình Huấn Luyện Chưng Cất Đa Thầy (`KDTrainer` — `src/kd_trainer.py`)

Mô hình `CEDSingleMoESplitFFN` được huấn luyện trong `15 epochs` trên tập dữ liệu `train.npy` ($30,240$ mẫu) với sự dẫn dắt đồng thời của 2 giáo viên **BEATs 12L (`90.34M` tham số)**:
- **`Teacher Fold0` ($T_0$):** Chuyên gia kháng nhiễu mạnh ở vùng SNR thấp/trung bình ($-5\text{ dB} \to 10\text{ dB}$).
- **`Teacher EMA` ($T_{\text{ema}}$):** Chuyên gia lọc giả ảnh FRCRN với trung bình trọng số mũ EMA ($\beta = 0.9995$), vượt trội ở vùng SNR cao ($15\text{ dB} \to 20\text{ dB}$).

### 4.1. Consistent Mixup Đồng Bộ (`apply_mixup_with_teachers`)
Với xác suất $p = 0.8$ và hệ số $\lambda \sim \text{Beta}(0.4, 0.4)$, phép nội suy Mixup được áp dụng đồng thời lên sóng âm $\mathbf{x}$, nhãn thực $\mathbf{y}$, mức $\text{SNR}_{\text{dB}}$ và phân phối xác suất mềm của cả 2 Teachers ở nhiệt độ $T = 2.0$:
$$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j, \quad \tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda)\mathbf{y}_j, \quad \widetilde{\text{SNR}} = \lambda \text{SNR}_i + (1-\lambda)\text{SNR}_j$$
$$\tilde{\mathbf{p}}_{T_0}^{(T)} = \lambda \text{Softmax}(\mathbf{z}_{T_0, i}/T) + (1-\lambda)\text{Softmax}(\mathbf{z}_{T_0, j}/T)$$
$$\tilde{\mathbf{p}}_{T_{\text{ema}}}^{(T)} = \lambda \text{Softmax}(\mathbf{z}_{T_{\text{ema}}, i}/T) + (1-\lambda)\text{Softmax}(\mathbf{z}_{T_{\text{ema}}, j}/T)$$
Nhãn thực sau Mixup được làm mịn (Label Smoothing $\epsilon = 0.05$): $\tilde{\mathbf{y}}_{\text{smooth}} = 0.95\,\tilde{\mathbf{y}} + \frac{0.05}{36}$.

### 4.2. Nhãn Chế Độ Nhiễu Mềm & Tổ Hợp Hàm Mất Mát 4 Thành Phần
Từ mức $\widetilde{\text{SNR}}$ liên tục (chỉ dùng trong lúc train), hệ thống tính nhãn định tuyến mục tiêu mềm $w_{\text{snr}} \in (0, 1)$ có tâm chuyển tiếp tại $15.0\text{ dB}$:
$$w_{\text{snr}} = \sigma\!\left(\frac{\widetilde{\text{SNR}} - 15.0}{2.5}\right), \quad \tilde{\mathbf{p}}_{T, \text{blend}}^{(T)} = (1 - w_{\text{snr}})\,\tilde{\mathbf{p}}_{T_0}^{(T)} + w_{\text{snr}}\,\tilde{\mathbf{p}}_{T_{\text{ema}}}^{(T)}$$
*(Ví dụ: $\widetilde{\text{SNR}} = -5\text{ dB} \implies w_{\text{snr}} = 0.0003$; $\widetilde{\text{SNR}} = 10\text{ dB} \implies w_{\text{snr}} = 0.119$; $\widetilde{\text{SNR}} = 15\text{ dB} \implies w_{\text{snr}} = 0.500$; $\widetilde{\text{SNR}} = 20\text{ dB} \implies w_{\text{snr}} = 0.881$).*

Hàm mục tiêu huấn luyện gồm 4 số hạng:
1. **Supervised Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$):**
   $$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{36} \tilde{y}_{\text{smooth}, c} \ln \text{Softmax}(\mathbf{z}_{\text{student}})_c$$
2. **Fused Dual-Teacher KD Loss ($\mathcal{L}_{\text{KD, fused}}$):**
   $$\mathcal{L}_{\text{KD, fused}} = -T^2 \sum_{c=1}^{36} \tilde{p}_{T, \text{blend}, c}^{(T)} \ln \text{Softmax}(\mathbf{z}_{\text{student}} / T)_c$$
3. **Regime-Specialized Expert KD Loss ($\mathcal{L}_{\text{KD, experts}}$):** Định hướng trực tiếp nhánh `Low-Expert` ($\mathbf{z}_{\text{low}}$) học từ `Teacher Fold0` và nhánh `High-Expert` ($\mathbf{z}_{\text{high}}$) học từ `Teacher EMA`:
   $$\mathcal{L}_{\text{KD, experts}} = (1 - w_{\text{snr}})\left[-T^2\sum_{c=1}^{36}\tilde{p}_{T_0, c}^{(T)}\ln\text{Softmax}(\mathbf{z}_{\text{low}}/T)_c\right] + w_{\text{snr}}\left[-T^2\sum_{c=1}^{36}\tilde{p}_{T_{\text{ema}}, c}^{(T)}\ln\text{Softmax}(\mathbf{z}_{\text{high}}/T)_c\right]$$
   $$\mathcal{L}_{\text{KD}} = 0.65\,\mathcal{L}_{\text{KD, fused}} + 0.35\,\mathcal{L}_{\text{KD, experts}}$$
4. **Router Supervision Loss ($\mathcal{L}_{\text{route}}$):**
   $$\mathcal{L}_{\text{route}} = \text{BCE}(\alpha, w_{\text{snr}}) = -\Big[w_{\text{snr}}\ln\alpha + (1 - w_{\text{snr}})\ln(1 - \alpha)\Big]$$

**Hàm mất mát tổng hợp** ($\lambda_{\text{kd}} = 0.65, \lambda_{\text{route}} = 0.20$):
$$\mathcal{L}_{\text{total}} = 0.35\,\mathcal{L}_{\text{CE}} + 0.65\,\mathcal{L}_{\text{KD}} + 0.20\,\mathcal{L}_{\text{route}}$$

### 4.3. Cấu Hình Tối Ưu Hóa Phân Tầng (Differential Learning Rates)
Sử dụng bộ tối ưu **AdamW** (`weight_decay = 0.05`, cắt chuẩn gradient `max_norm = 1.0`, lịch trình Cosine Annealing với `2 epochs` Linear Warmup) chia làm 3 nhóm tốc độ học:
- `backbone_params` (`Blocks 0..9` + `BiasAttention`): `lr = 4.0e-5` (giữ ổn định tri thức tiền huấn luyện AudioSet).
- `expert_params` (`ffn_shared`, `ffn_low`, `ffn_high`, `head_low`, `head_high`): `lr = 1.2e-4`.
- `router_params` (`SelfContainedSpectralTokenRouter`): `lr = 2.0e-4`.

---

## 5. Đặc Tả Cơ Chế Suy Luận `ChampionPipelineV2` (Zero-Oracle Dynamic Blending — `src/pipeline.py`)

Khi thực thi suy luận (`mode="self_contained_champion"`), pipeline **hoàn toàn không nhận `snr_db`** mà xử lý tự động theo 3 giai đoạn:

### Giai đoạn 1: 3-Way Temporal Test-Time Augmentation (TTA $\pm 50\text{ ms}$)
Với mỗi mẫu âm thanh $\mathbf{w} \in \mathbb{R}^{B \times 64000}$ ($16\text{ kHz}$), tạo 3 biến thể dịch thời gian bằng phép đệm phản xạ (`mode="reflect"`, $\Delta = \pm 800\text{ mẫu} = \pm 50\text{ ms}$):
$$\mathbf{w}_0 = \mathbf{w}, \quad \mathbf{w}_{+50} = \text{Pad}_{\text{reflect}}(\mathbf{w}_{:, 800:}, (0, 800)), \quad \mathbf{w}_{-50} = \text{Pad}_{\text{reflect}}(\mathbf{w}_{:, :-800}, (800, 0))$$
Với mỗi mô hình $m \in \{\text{CED-SplitFFN}, T_{\text{fold0}}, T_{\text{ema}}\}$, tính trung bình gia quyền logits:
$$\bar{\mathbf{z}}_m = \frac{1.0 \cdot \mathbf{z}_m(\mathbf{w}_0) + 0.5 \cdot \mathbf{z}_m(\mathbf{w}_{+50}) + 0.5 \cdot \mathbf{z}_m(\mathbf{w}_{-50})}{2.0} \in \mathbb{R}^{B \times 36}$$
Đồng thời trích xuất trọng số định tuyến nội tại $\alpha \in (0, 1)^{B \times 1}$ từ lượt chạy $\mathbf{w}_0$ của `CED-SplitFFN`.

### Giai đoạn 2: Hiệu Chuẩn Nhiệt Độ Bất Đối Xứng (Asymmetric Temperature Scaling)
Chuyển đổi logits sang xác suất đã hiệu chuẩn độ tự tin ($T_s = 1.40$ cho Student, $T_t = 1.20$ cho 2 Teachers):
$$\mathbf{P}_{\text{student}} = \text{Softmax}\!\left(\frac{\bar{\mathbf{z}}_{\text{CED-SplitFFN}}}{1.40}\right), \quad \mathbf{P}_{T_0} = \text{Softmax}\!\left(\frac{\bar{\mathbf{z}}_{T_0}}{1.20}\right), \quad \mathbf{P}_{T_{\text{ema}}} = \text{Softmax}\!\left(\frac{\bar{\mathbf{z}}_{T_{\text{ema}}}}{1.20}\right)$$

### Giai đoạn 3: Hòa Trộn Xác Suất Động Nội Tại Theo $\alpha$ (Zero-Oracle Blending)
Sử dụng trực tiếp trọng số $\alpha \in (0, 1)$ của bộ định tuyến nội tại để nội suy tuyến tính trọng số đóng góp của từng mô hình cho từng mẫu riêng biệt trong batch:
$$w_{\text{Student}} = 0.25$$
$$w_{T_0}(\alpha) = (1 - \alpha) \cdot 0.45 + \alpha \cdot 0.30$$
$$w_{T_{\text{ema}}}(\alpha) = (1 - \alpha) \cdot 0.30 + \alpha \cdot 0.45$$
$$\mathbf{P}_{\text{final}} = 0.25\,\mathbf{P}_{\text{student}} + w_{T_0}(\alpha)\,\mathbf{P}_{T_0} + w_{T_{\text{ema}}}(\alpha)\,\mathbf{P}_{T_{\text{ema}}}, \quad \hat{y} = \arg\max_{c \in \{1..36\}} P_{\text{final}, c}$$
- **Tính chất bảo toàn chuẩn xác suất:** $w_{\text{Student}} + w_{T_0}(\alpha) + w_{T_{\text{ema}}}(\alpha) = 0.25 + (1-\alpha)(0.75) + \alpha(0.75) \equiv 1.00$ với mọi $\alpha \in [0, 1]$.
- **Cơ chế thích nghi vật lý:** Khi $\alpha \to 0$ (môi trường nhiễu nặng $-5..5\text{ dB}$), `Teacher Fold0` giữ vai trò trụ cột ($w_{T_0} \to 0.45$). Khi $\alpha$ tăng lên ở vùng chuyển tiếp và SNR cao ($15..20\text{ dB}$), pipeline tự động dịch chuyển trọng số sang `Teacher EMA` ($w_{T_{\text{ema}}} \to 0.45$) mà không xảy ra hiện tượng đứt gãy bậc thang như ngưỡng cắt cứng $18\text{ dB}$.

---

## 6. Kết Quả Thực Nghiệm Chi Tiết (Trên Tập Kiểm Định Sạch 3,064 Mẫu)

| Chỉ số / Phân vùng SNR | `CED-SingleMoE-SplitFFN` (Standalone 1-Backbone `6.62M`, Không kèm Teacher lúc Test) | `Champion Pipeline v2: SplitFFN + Teachers (Oracle SNR = 18 dB)` | 🏆 `Champion Pipeline v2: SplitFFN + Teachers (ZERO ORACLE)` | Đối chiếu với `imp/moe` cũ (Dual-Student `10.84M` + Oracle SNR) |
| :--- | :---: | :---: | :---: | :---: |
| **Số lượng Student Backbone** | **1 Model (`6.62M`)** | **1 Model (`6.62M`)** | **1 Model (`6.62M`)** | 2 Models (`10.84M`) |
| **Cần nhãn `snr_db` lúc Test** | **Không (`Zero-Oracle`)** | Có (Oracle $\ge 18\text{ dB}$) | **Không (`Zero-Oracle`)** | Có (Oracle $\ge 18\text{ dB}$) |
| **SNR = $-5\text{ dB}$ (500 mẫu)** | `68.10%` | `72.57%` | **`72.57%`** | `72.20%` (`+0.37%`) |
| **SNR = $0\text{ dB}$ (518 mẫu)** | `65.36%` | `72.99%` | **`72.41%`** | `71.62%` (`+0.79%`) |
| **SNR = $5\text{ dB}$ (505 mẫu)** | `60.66%` | `64.55%` | **`65.37%`** | `64.55%` (`+0.82%`) |
| **SNR = $10\text{ dB}$ (513 mẫu)** | `58.87%` | `63.94%` | **`64.52%`** | `62.96%` (`+1.56%`) |
| **SNR = $15\text{ dB}$ (519 mẫu)** | `51.68%` | `56.21%` | **`57.59%`** | `57.79%` (`-0.20%`) |
| **SNR = $20\text{ dB}$ (509 mẫu)** | `46.56%` | `51.08%` | **`50.69%`** | `51.87%` |
| **All Top-1 Accuracy** | `58.62%` | `63.64%` | **`63.94%`** | `63.58%` (**`+0.36%`**) |
| **All Top-3 Accuracy** | `76.63%` | `79.77%` | **`79.80%`** | `80.19%` |
| **Balanced Accuracy (Macro)** | `56.71%` | `62.02%` | **`62.24%`** | `62.03%` (**`+0.21%`**) |
| **Macro mAP** | `63.54%` | `70.10%` | **`70.05%`** | `70.08%` (**Đạt chuẩn $\ge 70\%$**) |
| **Macro AUC** | `95.28%` | `96.16%` | **`96.10%`** | `96.19%` |

> **Điểm nhấn học thuật quan trọng:** Phiên bản **Zero-Oracle (`63.94%` Top-1)** không những loại bỏ hoàn toàn việc gian lận nhãn `snr_db` lúc suy luận mà còn đạt độ chính xác tổng thể **cao hơn cả phiên bản dùng Oracle SNR cắt cứng tại `18 dB` (`63.64%` Top-1)**, đặc biệt bứt phá mạnh ở các mốc chuyển tiếp `5 dB` (`+0.82%`), `10 dB` (`+0.58%`) và `15 dB` (`+1.38%` so với Oracle cắt cứng) nhờ cơ chế nội suy mềm liên tục theo $\alpha$.

---

## 7. Giao Diện Dòng Lệnh (CLI) & Tích Hợp Mã Nguồn

### 7.1. Chạy đánh giá trực tiếp qua CLI (`champion_pipeline.py`)
```bash
# Đánh giá Champion Pipeline v2: SplitFFN + Teachers (Chế độ mặc định: Zero-Oracle Self-Contained)
python imp/moe-v2/champion_pipeline.py \
  --student-type split_ffn \
  --mode self_contained \
  --student-ckpt /marimo/moe-v2/checkpoints/ced-singlemoe-splitffn-kd/best_model.pt \
  --teacher-fold0 /marimo/pretrained/beats_teacher_fold0_weights.pt \
  --teacher-ema /marimo/pretrained/beats_teacher_ema_standalone.pt \
  --data-root /marimo/datasets/mix-dataset-v2 \
  --cache-dir /marimo/datasets

# Chạy chế độ độc lập chỉ dùng 1 model Student duy nhất (6.62M params, không nạp Teachers)
python imp/moe-v2/champion_pipeline.py --student-type split_ffn --mode standalone
```

### 7.2. Sử dụng qua Python API (`src/pipeline.py`)
```python
import torch
from src.pipeline import load_pipeline_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = load_pipeline_v2(
    student_type="split_ffn",
    student_ckpt="/marimo/moe-v2/checkpoints/ced-singlemoe-splitffn-kd/best_model.pt",
    teacher_fold0_ckpt="/marimo/pretrained/beats_teacher_fold0_weights.pt",
    teacher_ema_ckpt="/marimo/pretrained/beats_teacher_ema_standalone.pt",
    beat_config_path="config/beat.yaml",
    device=device,
)

# Suy luận Zero-Oracle trực tiếp từ batch âm thanh nhiễu ước lượng [B, 64000] (không cần truyền snrs_db)
probs = pipeline.predict_proba(wave, mode="self_contained_champion")
pred_labels = pipeline.predict(wave, mode="self_contained_champion")
```

---

## 8. Tài Liệu Tham Khảo Chuẩn Xác Của Phương Pháp

1. **V-MoE (Last-K Split-FFN):** C. Riquelme et al., *"Scaling Vision with Sparse Mixture of Experts,"* in **NeurIPS**, 2021. [arXiv:2106.05974](https://arxiv.org/abs/2106.05974).
2. **DeepSeekMoE (Shared + Routed Experts):** D. Dai et al., *"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models,"* in **ACL**, 2024. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
3. **Soft MoE (Continuous Soft Gating):** J. Puigcerver et al., *"From Sparse to Soft Mixtures of Experts,"* in **ICLR**, 2024. [arXiv:2308.00951](https://arxiv.org/abs/2308.00951).
4. **CED (AudioSet ViT Backbone & Consistent KD):** H. Dinkel et al., *"CED: Consistent Ensemble Distillation for Audio Tagging,"* in **ICASSP**, 2024. [arXiv:2308.11957](https://arxiv.org/abs/2308.11957).
5. **BEATs (2D Gated Relative Position Bias & Teacher Architecture):** S. Chen et al., *"BEATs: Audio Pre-Training with Acoustic Tokenizers,"* in **ICML**, 2023. [arXiv:2212.09058](https://arxiv.org/abs/2212.09058).
6. **Consistent Teaching Mixup KD:** L. Beyer et al., *"Knowledge Distillation: A Good Teacher is Patient and Consistent,"* in **CVPR**, 2022. [arXiv:2106.05237](https://arxiv.org/abs/2106.05237).
