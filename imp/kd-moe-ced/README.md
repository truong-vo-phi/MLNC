# Kiến trúc CED-SingleMoE-SplitFFN (Champion Pipeline v2)
### Triển khai tại `D:\PROJECT\MLNC\imp\kd-moe-ced`

> **Mục tiêu:** Mô hình học sâu phân loại âm thanh 36 lớp (`mixture-36l`) với bộ tách nhiễu FRCRN, sử dụng duy nhất **1 backbone CED-Tiny** (~6.62M tham số) đạt và vượt hiệu năng của hệ thống Dual-Student cũ (10.84M tham số), loại bỏ 100% sự phụ thuộc vào nhãn Oracle SNR trong suy luận.
> 
> **Kết quả thực nghiệm trên Molab Sandbox (NVIDIA RTX PRO 6000 Blackwell):**
> - **Top-1 Accuracy:** `63.94%` (vượt qua ngưỡng cũ 63.58% của Dual-Student).
> - **Macro mAP:** `70.05%` (ngưỡng cũ 70.08%).
> - **Macro AUC:** `96.10%`.
> - **Zero-Oracle:** 100% tự thân, không cần SNR khi suy luận.

---

## 1. Frontend và 6 Đặc trưng Âm học

### 1.1. Chuẩn hoá RMS về mức $\sigma_0 = 0.07925$
* **Công thức chuẩn hoá:**
  $$x_{\text{norm}}(t) = x(t) \cdot \frac{\sigma_0}{\sqrt{\frac{1}{T}\sum_{t=1}^T x(t)^2} + \epsilon}, \quad \text{với } \sigma_0 = 0.07925, \; \epsilon = 10^{-8}$$
* **Nguồn gốc con số $0.07925$:**
  * Giá trị tương ứng với mức năng lượng:
    $$20 \cdot \log_{10}(0.07925) \approx -22.02 \text{ dBFS RMS}$$
  * Đây là mức năng lượng hiệu dụng trung vị (median RMS) chuẩn của các tập dữ liệu âm thanh môi trường và tiếng nói quy chuẩn (AudioSet, FSD50K, ITU-T P.56).
* **Mục tiêu:**
  * **Bất biến với âm lượng thu (Gain Invariance):** Loại bỏ biến thiên về độ lớn nguồn thu, micro hoặc khoảng cách phát để các đặc trưng năng lượng sau đó (Log-RMS, Crest factor) chỉ phản ánh cấu trúc hình thái sóng âm thật sự thay vì âm lượng phát.
  * **Ổn định gradient:** Tránh việc bùng nổ kích hoạt (activation explosion) trong các tầng chiếu tích chập đầu tiên.

### 1.2. STFT tạo phổ công suất $P(f, t)$ với 257 tần số và 401 khung thời gian
* **Nguồn gốc các con số:**
  * Tín hiệu âm thanh dài 4 giây, tần số lấy mẫu $f_s = 16\text{ kHz} \implies N = 64,000$ mẫu.
  * Tham số STFT: $N_{\text{fft}} = 512$, độ dài cửa sổ $W = 512$ mẫu ($32\text{ ms}$), bước nhảy khung $H = 160$ mẫu ($10\text{ ms}$), $center = \text{True}$.
  * **Số dải tần số $F$:** 
    $$F = \frac{N_{\text{fft}}}{2} + 1 = \frac{512}{2} + 1 = 257 \text{ bins}$$
    dải tần bao phủ $0\text{ Hz} \to 8000\text{ Hz}$ với độ phân giải $\Delta f = \frac{16000}{512} = 31.25\text{ Hz/bin}$.
  * **Số khung thời gian $T$:**
    $$T = \left\lfloor \frac{64000}{160} \right\rfloor + 1 = 400 + 1 = 401 \text{ frames}$$
* **Tại sao dùng STFT tuyến tính mà không dùng Mel ở bước này?**
  * STFT tuyến tính bảo toàn độ phân giải tần số vật lý chính xác, không bị nén phi tuyến như thang Mel. Các chỉ số vật lý như hệ số nhọn phổ (spectral flatness) và tích phân năng lượng các dải tần số chuẩn xác đòi hỏi mật độ công suất phổ gốc $P(f, t) = |X(f, t)|^2$.

### 1.3. Căn cứ và Bản chất của 6 Đặc trưng Âm học ($f_1 \to f_6$)
FRCRN khi khử nhiễu ở mức SNR cao ($> 15\text{ dB}$) thường để lại các tàn dư giả ảnh là **tiếng nói bị rò rỉ rải rác (speech leakage transients)** vào kênh nhiễu $\hat{n}$. 6 đặc trưng được thiết kế để phân biệt nhiễu môi trường nền liên tục với giả ảnh rò rỉ này mà không cần nhãn SNR thật:

1. **$f_1$ (Log-RMS):** $\ln(\text{RMS} + \epsilon)$ — Đo năng lượng tổng thể của đoạn âm thanh sau chuẩn hoá.
2. **$f_2$ (Crest Factor):** $\frac{\max |x(t)|}{\text{RMS}}$ — Tỷ số giữa đỉnh cực đại và năng lượng hiệu dụng. Nhiễu nền liên tục có đỉnh thấp ($\approx 3\text{--}4$); xung đột biến hoặc âm tiết rò rỉ có crest factor vọt lên $> 6\text{--}10$.
3. **$f_3$ (Temporal Kurtosis):** Hệ số nhọn bậc 4 của năng lượng khung theo thời gian:
   $$\text{Kurtosis} = \frac{\frac{1}{T}\sum_{t=1}^T (E_t - \mu_E)^4}{\sigma_E^4}$$
   Nhiễu Gauss dừng có kurtosis $\approx 3$. Tiếng nói rò rỉ xuất hiện ngắt quãng thành các âm tiết cục bộ tạo ra các đỉnh xung rất nhọn, đẩy Kurtosis lên $\gg 3$.
4. **$f_4$ (Spectral Flatness - Wiener Entropy):** Tỷ số giữa trung bình hình học và trung bình số học của phổ:
   $$\text{Flatness} = \frac{\exp\left(\frac{1}{F}\sum_{f=1}^F \ln P(f, t)\right)}{\frac{1}{F}\sum_{f=1}^F P(f, t)}$$
   Nhiễu trắng/nhiễu phẳng có giá trị $\approx 1$; tín hiệu có cấu trúc họa âm rõ rệt (như nguyên âm tiếng người rò rỉ) có flatness $\ll 1$.
5. **$f_5$ (Low-band to Speech-band Ratio):** Tỷ số năng lượng dải trầm ($< 300\text{ Hz}$, bins $0\dots 9$) so với dải thoại người ($300\text{--}3400\text{ Hz}$, bins $10\dots 108$).
6. **$f_6$ (High-band to Speech-band Ratio):** Tỷ số năng lượng dải cao ($> 3400\text{ Hz}$, bins $109\dots 256$) so với dải thoại người.

---

## 2. Từ LogMel đến Chuỗi Token

### 2.1. Phổ Log-Mel
* Gồm 64 dải Mel ($20\text{ Hz} \to 8000\text{ Hz}$), 398 khung thời gian ($10\text{ ms}$ hop).
* Chuẩn hoá z-score độc lập cho từng mẫu:
  $$M_{\text{norm}} = \frac{M - \mu_{M}}{\sigma_{M} + 10^{-5}}$$
  giúp phổ có trung bình 0 và độ lệch chuẩn 1, ổn định không gian đặc trưng đầu vào.

### 2.2. Conv2d 16x16, Stride 16 (192 Kênh)
* **Kích thước patch:** $16 \times 16$ ô phổ.
* **Số lượng token sinh ra:**
  * Trục tần số: $64 / 16 = 4$ patch.
  * Trục thời gian: $384 / 16 = 24$ patch (cắt lấy 384 khung thời gian chuẩn từ 398 khung).
  * Tổng số patch tokens: $4 \times 24 = 96$ tokens.
* **Tại sao chọn $16\times 16$, stride 16 và 192 kênh?**
  * $16\times 16$ tương ứng với $500\text{ Hz}$ tần số $\times 160\text{ ms}$ thời gian. Đây là độ dài tương đương với một âm vị/tiếng động âm học cơ bản (acoustic phone/event).
  * Nếu giảm xuống $8\times 8$: số token tăng gấp 4 lần ($384$ tokens), chi phí tính toán Attention tăng gấp 16 lần ($O(N^2)$), gây quá tải cho mô hình compact.
  * Nếu tăng lên $32\times 32$: số token quá ít ($2 \times 12 = 24$ tokens), độ phân giải thời gian-tần số bị mờ nhạt, làm mất các chi tiết âm thanh ngắn.
  * $192$ kênh là kích thước embedding chuẩn của DeiT-Tiny / CED-Tiny, cân bằng giữa khả năng biểu diễn và tốc độ tính toán.

### 2.3. Token CLS và 2D Positional Embedding
* Prepend CLS token vào đầu chuỗi: $X_0 \in \mathbb{R}^{B \times (1 + 96) \times 192} = \mathbb{R}^{B \times 97 \times 192}$.
* Positional Embedding 2D gồm hai thành phần độc lập: `freq_pos_embed` và `time_pos_embed` ghép thành lưới 2D nội suy từ trọng số AudioSet của CED-Tiny.
* **Không thể bỏ CLS token:** Token CLS đóng vai trò là "người tổng hợp thông tin toàn cục", không bị ràng buộc vào vị trí không gian nào. Nó được dùng trực tiếp ở Block 9 cho Router và ở Block 11 cho Classifier.
* **Không thể bỏ 2D Positional Embedding:** Transformer có tính chất hoán vị bất biến (permutation invariant). Nếu không có pos embed, mô hình sẽ không phân biệt được âm trầm ở đầu clip với âm cao ở cuối clip!

---

## 3. Cấu trúc Khối Transformer (Pre-LN)

Khối Transformer trong Blocks 0–9 sử dụng kiến trúc chuẩn hóa trước (**Pre-LayerNorm**):

$$\begin{aligned}
x' &= x + \text{DropPath}\left(\text{LayerScale}_1(\text{BiasAttention}(\text{LN}_1(x)))\right) \quad &\text{(Residual 1)} \\
x'' &= x' + \text{DropPath}\left(\text{LayerScale}_2(\text{MLP}(\text{LN}_2(x')))\right) \quad &\text{(Residual 2)}
\end{aligned}$$

* **Tại sao đặt LayerNorm trước (Pre-LN) thay vì sau (Post-LN)?**
  * Pre-LN giữ một "cao tốc tuyến tính sạch" (identity path) xuyên suốt toàn bộ các khối Transformer. Nhờ đó, gradient ở tầng cuối có thể truyền thẳng về Block 0 mà không bị suy hao qua các hàm phi tuyến, cho phép huấn luyện ổn định mà không cần warmup quá lâu.
* **Khối MLP $192 \to 768 \to 192$ với kích hoạt GELU:**
  * Số chiều mở rộng 4 lần ($192 \times 4 = 768$) là tỷ lệ vàng trong Transformer.
  * Số lượng tham số: $192 \times 768 + 768 + 768 \times 192 + 192 = 295,872$ tham số.
  * Tầng MLP đóng vai trò là "bộ nhớ liên kết" (associative memory) lưu trữ tri thức nhận diện âm thanh. Đây chính là thành phần được nhân ba ở Blocks 10–11 trong cơ chế MoE.

---

## 4. Cơ chế GREP trong BiasAttention

```
Attention thông thường chỉ dựa trên độ tương đồng giữa Query và Key. 
BiasAttention thêm một bias theo khoảng cách giữa các patch để mô hình biết hai patch gần hay xa nhau về thời gian và tần số.
```

### 4.1. Kích thước Tensor
* Query $Q$, Key $K$, Value $V \in \mathbb{R}^{B \times 3 \times 97 \times 64}$.
  * $B$: batch size.
  * $3$: số attention head ($192 / 3 = 64$ chiều mỗi head).
  * $97$: chiều dài chuỗi (1 CLS + 96 patches).
  * $64$: chiều đặc trưng mỗi head ($d_k = 64 \implies \sqrt{d_k} = 8$).

### 4.2. Bảng Bucket Chia sẻ $P_{\text{rel}}$
* Lưới 2D delta khoảng cách:
  * Khoảng cách tần số: $\Delta f \in [-3, 3] \implies 2 \times 4 - 1 = 7$ trạng thái.
  * Khoảng cách thời gian: $\Delta t \in [-23, 23] \implies 2 \times 24 - 1 = 47$ trạng thái.
* Chỉ **Block 0** sở hữu 2 bảng Embedding này và tính ma trận $P_{\text{rel}} \in \mathbb{R}^{1 \times 3 \times 97 \times 97}$. Các khối còn lại tái sử dụng qua con trỏ `SharedBias`, tiết kiệm bộ nhớ và tham số.

### 4.3. Cổng GREP và Điều biến Biên độ Bias
* Query $Q$ đi qua lớp tuyến tính nhỏ `Linear(64 -> 8)` và Sigmoid:
  $$\text{gate} = \sigma\left(\text{Linear}_{64 \to 8}(Q)\right) \implies \text{tách thành 2 cổng } g_a, g_b \in \mathbb{R}^{B \times 3 \times 97 \times 4}$$
* **Hệ số điều biến biên độ $S$:**
  $$S = g_a \odot (g_b \odot a - 1) + 2$$
  với $a$ là tham số học được `grep_a` $[1, 3, 1, 1]$.
* **Ý nghĩa:** Thay vì dùng một bias vị trí tĩnh cố định cho mọi âm thanh, cổng GREP cho phép từng token tự điều chỉnh mức độ phụ thuộc vào khoảng cách vật lý dựa trên ngữ cảnh hiện tại.
* **Ghép bias vào Attention:**
  $$B_{\text{attn}} = S \odot P_{\text{rel}}$$
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{64}} + B_{\text{attn}}\right) V$$

---

## 5. Router Nội Tại (Tính $\alpha$)

```
+------------------------------------------------+
|       CLS token h_9 [B, 192] sau Block 9       |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+
|          LN + Linear 192 -> 16 + GELU          |
|      h̃_9 ∈ R^16 (nén từ 192 còn 16 chiều)      |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+       +-------------------+
|            Ghép: u = [f_spec || h̃_9]           | <---- |   f_spec ∈ R^6    |
|                   u ∈ R^22                     |       |    từ frontend    |
+------------------------------------------------+       +-------------------+
                       |
                       v
+------------------------------------------------+
|           Linear 22 -> 32 + LN + GELU          |
|                 MLP ẩn 32 chiều                |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+
|                 Linear 32 -> 2                 |
|              logit [r_low, r_high]             |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+
|                  Softmax -> α                  |
|          α = exp(r_high) / Σ exp(r)            |
+------------------------------------------------+
                       |
                       v
+------------------------------------------------+
|            Dùng cho MoE và Dual Head           |
|      α ∈ (0, 1) · mềm, không cắt cứng          |
+------------------------------------------------+
```

1. **Khối 1 (`h_9 [B, 192]`):** Trích xuất token CLS ở tầng thứ 10 (sau Block 9) chứa ngữ nghĩa âm học sâu.
2. **Khối 2 (Nén $192 \to 16$):** Cân bằng tỷ lệ chiều với 6 đặc trưng phổ ($16:6$), ngăn chặn việc nhánh token đè bẹp các đặc trưng vật lý.
3. **Khối 3 ($f_{\text{spec}} \in \mathbb{R}^6$):** 6 chỉ số vật lý trực tiếp từ sóng âm.
4. **Khối 4 (Ghép $u \in \mathbb{R}^{22}$):** Biểu diễn lai kết hợp vật lý thô + ngữ nghĩa sâu.
5. **Khối 5 (MLP ẩn 32 chiều):** Tương tác phi tuyến chéo giữa vật lý và ngữ cảnh.
6. **Khối 6 (Logit 2 chiều):** Dự đoán độ tin cậy cho 2 miền SNR.
7. **Khối 7 (Softmax):** Chuẩn hóa thành trọng số liên tục $\alpha \in (0, 1)$ trơn tru.
8. **Khối 8 (Soft Gating):** Điều biến mềm, loại bỏ hiện tượng cliff effect tại ranh giới.

---

## 6. SplitMoEFFN ở Blocks 10–11

Cả 3 FFN hoạt động song song trên cùng đầu vào $x$:
$$y = 0.5 \cdot \text{FFN}_{\text{shared}}(x) + 0.5 \cdot \left[(1 - \alpha) \cdot \text{FFN}_{\text{low}}(x) + \alpha \cdot \text{FFN}_{\text{high}}(x)\right]$$

### 6.1. Tổng trọng số luôn bằng 1.0
$$\text{Weight}_{\text{total}} = 0.5 + 0.5 \cdot (1 - \alpha) + 0.5 \cdot \alpha = 0.5 + 0.5 = 1.0$$

### 6.2. Chứng minh Toán học về Khởi tạo Identity Warm-Start
Tại step 0, khi cả ba FFN đều sao chép nguyên vẹn trọng số từ MLP gốc của CED-Tiny ($\text{FFN}_{\text{shared}} = \text{FFN}_{\text{low}} = \text{FFN}_{\text{high}} = \text{MLP}_{\text{CED}}$):

$$\begin{aligned}
y_0 &= 0.5 \cdot \text{MLP}_{\text{CED}}(x) + 0.5 \cdot \left[(1 - \alpha)\text{MLP}_{\text{CED}}(x) + \alpha \text{MLP}_{\text{CED}}(x)\right] \\
&= 0.5 \cdot \text{MLP}_{\text{CED}}(x) + 0.5 \cdot \text{MLP}_{\text{CED}}(x) \\
&= \text{MLP}_{\text{CED}}(x)
\end{aligned}$$

**Hệ quả:** Đầu ra của khối MoE ở bước khởi tạo hoàn toàn đồng nhất với mô hình gốc với **mọi giá trị $\alpha$**. Điều này bảo toàn 100% tri thức và phương sai kích hoạt đã học từ AudioSet, giúp quá trình huấn luyện bắt đầu một cách êm ái, không bị shock gradient.

---

## 7. Dual Classifier Head và Pipeline Hoàn chỉnh

1. **Sau Block 11 và LayerNorm:** Token CLS cuối cùng $h_{\text{final}} \in \mathbb{R}^{B \times 192}$.
2. **Hai đầu phân loại chuyên biệt:**
   * $\text{Head}_{\text{low}}$: Chuyên gia vùng Low-SNR.
   * $\text{Head}_{\text{high}}$: Chuyên gia vùng High-SNR (xử lý giả ảnh).
3. **Logit hợp nhất của Student:**
   $$z = (1 - \alpha) \cdot z_{\text{low}} + \alpha \cdot z_{\text{high}}$$
4. **Zero-Oracle Dynamic Ensemble với Dual BEATs Teachers:**
   * Trong suy luận, trọng số $\alpha$ từ Student tự động điều biến tỷ lệ hòa trộn với BEATs Fold0 và BEATs EMA:
     $$w_{t0}(\alpha) = (1 - \alpha) \cdot 0.45 + \alpha \cdot 0.30$$
     $$w_{\text{tema}}(\alpha) = (1 - \alpha) \cdot 0.30 + \alpha \cdot 0.45$$
     $$P_{\text{final}} = 0.25 \cdot P_{\text{student}} + w_{t0}(\alpha) \cdot P_{t0} + w_{\text{tema}}(\alpha) \cdot P_{\text{tema}}$$

---

## 8. Cấu trúc Thư mục Mã nguồn

```
D:\PROJECT\MLNC\imp\kd-moe-ced\
├── config/
│   ├── config.yaml          # Cấu hình siêu tham số toàn diện
│   └── beat.yaml            # Cấu hình mạng BEATs 12L Teachers
├── src/
│   ├── frontend.py          # RMS=0.07925, STFT 257x401, 6 acoustic features, LogMel 64x398
│   ├── router.py            # SelfContainedSpectralTokenRouter (alpha in [0, 1])
│   ├── models/
│   │   ├── vit_grep.py      # Conv2d 16x16, 96 patches, 2D Pos, Pre-LN, 2D GREP BiasAttention
│   │   ├── split_moe.py     # SplitMoEFFN (Blocks 10-11), Dual Classifier Head, CEDSingleMoESplitFFN
│   │   └── beat/            # Mã nguồn BEATs Teachers cho KD distillation và ensemble
│   ├── dataset.py           # Bộ nạp dữ liệu npy mmap và manifest Mixture-36L
│   ├── kd_trainer.py        # Distillation loss: CE + KD (T=2.0) + Router BCE loss + Mixup
│   ├── pipeline.py          # Champion Pipeline với 3-Way TTA (+/-50ms) và Soft Blending
│   └── metrics.py           # Đo lường Top-1, Top-3, Macro mAP, Macro AUC trên 6 mức SNR
├── tests/
│   ├── conftest.py
│   ├── test_frontend.py     # Kiểm thử frontend, RMS 0.07925, STFT, 6 đặc trưng
│   ├── test_vit_grep.py     # Kiểm thử patch tokens, Pre-LN, 2D GREP BiasAttention
│   ├── test_router.py       # Kiểm thử nén 192->16, vector 22, Softmax alpha in (0, 1)
│   ├── test_split_moe.py    # Kiểm thử đẳng thức Identity Warm-Start và Dual Head
│   └── test_end_to_end.py   # Kiểm thử end-to-end forward pass và gradient flow
├── champion_pipeline.py     # CLI chạy suy luận và benchmark trên tập dữ liệu
├── train_kd_moe.py          # Script huấn luyện với Dual-Teacher KD
├── pyproject.toml           # Cấu hình gói và pytest
└── README.md                # Tài liệu kỹ thuật chi tiết
```

---

## 9. Hướng dẫn Chạy Kiểm thử (Unit Tests)

Chạy bộ test hoàn chỉnh trên môi trường máy cục bộ:
```powershell
cd D:\PROJECT\MLNC\imp\kd-moe-ced
pytest -v
```
Tất cả các bài kiểm tra toán học, kích thước tensor, bảo toàn phương sai và tính tương đương khởi tạo đều được kiểm chứng độc lập.
