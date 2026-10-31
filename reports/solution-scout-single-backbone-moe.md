# Research Solution Scout: Tối Ưu Hóa Single-Backbone Mixture-of-Experts (MoE) Cho CED-Tiny

**Ngày thực hiện:** 2026-10-30  
**Phạm vi khảo sát:** Toàn bộ nhánh thực nghiệm trong `imp/` (`ced-tiny`, `ft-beat`, `ema-beat`, `kd-v3`, `kd-v4`, `kd-v5`, `kd-beat-ced`, `ban-with-beat`, `moe`) và đối chiếu với các công trình nghiên cứu quốc tế (ICLR, NeurIPS, ICASSP, INTERSPEECH).

---

## 1. Khuyến Nghị Điều Hành (Executive Recommendation)

Để rút gọn `ChampionPipeline` từ **2 mô hình Student CED-Tiny độc lập (`10.84M` tham số)** về **1 mô hình CED-Tiny duy nhất (`~5.6M – 6.0M` tham số)** mà vẫn giữ nguyên (hoặc vượt) khả năng chuyên biệt hóa Low-SNR / High-SNR và loại bỏ sự phụ thuộc vào Oracle SNR lúc suy luận, hướng đi tối ưu nhất là kiến trúc **Shared-Backbone Split-Top-K MoE kết hợp Self-Contained Spectral/Token Router và Dual-Teacher Regime Distillation**:

1. **Kiến trúc Single Backbone (`CED-SingleMoE`):**
   - **Chia sẻ hoàn toàn (Shared Stem & Early/Mid Blocks):** Giữ nguyên 100% `LogMel Frontend`, `PatchEmbed`, `2D Gated Relative Position Bias (BiasAttention)` ở cả 12 tầng, và toàn bộ các khối Transformer từ **Block 0 đến Block 9** (`10/12` blocks).
   - **Tách Chuyên gia ở Tầng Sâu (Split-FFN ở 2 Block cuối 10–11 + Dual/Tri-Head) hoặc HydraLoRA Experts:**
     - *Phương án A (Split-Top-2 FFN + Shared Expert - Khuyến nghị chính):* Tại Block 10, Block 11 và Classifier Head, sử dụng **1 Shared FFN/Head** (học biểu diễn chung cho mọi mức SNR) + **2 Routed Expert FFN/Head** (Chuyên gia Low-SNR và Chuyên gia High-SNR). Tất cả đều được khởi tạo trực tiếp từ trọng số pretrained của `mispeech/ced-tiny` (tránh hoàn toàn hiện tượng overfitting do khởi tạo ngẫu nhiên từng gặp ở `kd-v3`). Tổng tham số chỉ tăng **`+0.59M`** (`6.01M` so với `10.84M`, tiết kiệm **44.5% tham số**).
     - *Phương án B (HydraLoRA / Asymmetric LoRA Experts - Siêu gọn):* Giữ nguyên 100% `5.42M` tham số gốc của CED-Tiny làm khung xương sống dùng chung, gắn các nhánh **HydraLoRA** ($\Delta \mathbf{W}_k = \mathbf{B}_k \mathbf{A}$ với ma trận nén $\mathbf{A}$ dùng chung và $K$ ma trận mở rộng $\mathbf{B}_k$ chuyên biệt, rank $r=16$) vào tầng FFN của 4–6 block cuối kèm Multi-Head Classifier. Tổng tham số chỉ **`5.58M` (+2.9% tham số)**, chống overfitting cực tốt trên vùng `20 dB` (vốn chỉ chiếm $1/6$ dữ liệu).
2. **Tối ưu Bộ Định Tuyến Nội Tại (Internal Acoustic-Token Router — Không cần Oracle SNR hay `x_mix`):**
   - Thay vì dùng `EnergySNRRouter` cần cả `mixture` lẫn `noise` (hoặc truyền `snr_db` thật như lỗi `CRITICAL #3` trong `imp/moe/audit`), trích xuất **4 đặc trưng thống kê phổ/thời gian trực tiếp từ $\hat{\mathbf{n}}$** (Kurtosis năng lượng khung, Spectral Flatness, High-to-Low Band Ratio, Crest Factor) kết hợp với **trung bình token ẩn ở tầng giữa (Block 9)** đưa qua MLP nhẹ để dự đoán trọng số cổng mềm $g(\mathbf{x}) \in [0, 1]^E$.
   - Trong lúc **train**, giám sát trực tiếp bộ định tuyến bằng hàm mất mát phụ $\mathcal{L}_{\text{route}}$ dựa trên nhãn `snr_db` có sẵn trong manifest. Lúc **inference**, mô hình tự định tuyến 100% từ $\hat{\mathbf{n}}$.
3. **Chưng Cất Đa Thầy Theo Chế Độ Nhiễu (Regime-Aware Dual-Teacher KD):**
   - Trong một lượt huấn luyện duy nhất, nhánh Low-Expert được dẫn dắt mạnh bởi `Teacher Fold0` ($0.65 \cdot P_{\text{fold0}} + 0.35 \cdot P_{\text{ema}}$), nhánh High-Expert được dẫn dắt mạnh bởi `Teacher EMA` ($0.65 \cdot P_{\text{ema}} + 0.35 \cdot P_{\text{fold0}}$).

---

## 2. Định Khung Bài Toán & Tổng Hợp Các Phương Pháp Đã Triển Khai Trong `imp/`

### 2.1. Tổng kết thực nghiệm từ kho mã `imp/`

| Thư mục / Phương pháp | Kiến trúc Student | Tham số Student | Val Macro-Acc / Top-1 | Val mAP | Bài học cốt lõi rút ra |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `imp/kd-v3` & `src/model/kdmoe` | Custom Hybrid (Attn+BiGRU) + MaskGate + 4 MLP Experts / MobileViT-S | `1.25M – 6.38M` | `0.395` | `~0.41` | Khởi tạo từ đầu (random init) trên 30,240 mẫu với 4 MLP experts bị overfit nặng và không học được biểu diễn âm học nền tảng. |
| `imp/kd-v4` (T1, T2, E2–E8) | DeiT-Tiny (ImageNet) + ReviewKD; BEATs cắt 6L / 4L; ViT-S | `5.45M – 47.8M` | `0.380 – 0.577` | `0.372 – 0.630` | Pretrain ImageNet không khớp miền âm thanh; cắt lớp BEATs bị lệch hệ số $\alpha$ của DeepNorm và vẫn overfit (`train_acc > 0.94`, `gap > 0.36`). |
| `imp/kd-v5` & `imp/ced-tiny` | **CED-Tiny** (`ViTSBias`: 2D Gated RelPosBias + Time-Major + `dropout=0.0`) | **`5.42M`** | **`0.629`** | **`0.701`** | **Bước ngoặt:** Trọng số CED-Tiny AudioSet + Time-Major $(24 \times 4)$ + Gated Relative Bias + tắt Dropout (`std=12.5`) + Mixup `0.8` đạt cân bằng bias-variance tốt nhất (`gap = 0.223`). |
| `imp/kd-beat-ced` | CED-Tiny + Logit KD + Attention Transfer (AT) + BEATs Bias KD | `5.42M` | `~0.629` | `~0.701` | Trong `kd-v4`, ép MSE trực tiếp lên Q/K/V và position bias ở mọi tầng không mang lại cải thiện rõ rệt (`< 0.01`) so với Logit KD + Mixup. |
| `imp/ban-with-beat` | Born-Again Networks (Multi-generation KD ensemble) | `5.42M × K` | — | — | Cải thiện nhờ trung bình nhiều thế hệ nhưng làm bội số hóa chi phí suy luận theo số thế hệ $K$. |
| **`imp/moe` (`ChampionPipeline`)** | **2× CED-Tiny (`S1 Low` + `S2 High`)** + 2× BEATs Teachers + 3-Way TTA | **`10.84M`** (Student) | **`0.6358`** (Clean Val) | **`0.7008`** | Tách chuyên gia Low-SNR (`<18 dB`) và High-SNR (`>=18 dB`) giúp mức `20 dB` tăng vọt **`+6.88%`** (`51.87%`), nhưng tốn gấp đôi backbone (`2 × 5.42M`) và đang phụ thuộc vào ngưỡng cắt cứng / Oracle SNR. |

### 2.2. Tại sao tách Expert Low-SNR và High-SNR lại hiệu quả, nhưng nhân đôi Backbone là dư thừa?

1. **Bản chất vật lý của sự dịch chuyển miền (Covariate Shift) theo SNR sau FRCRN:**
   - Tín hiệu đầu vào của mô hình phân loại không phải là hỗn hợp gốc $\mathbf{x}_{\text{mix}}$, mà là phần dư nhiễu $\hat{\mathbf{n}} = \mathbf{x}_{\text{mix}} - \hat{\mathbf{s}}$ sau khi đi qua mạng tách tiếng nói FRCRN, rồi được chuẩn hóa năng lượng bằng `rms_normalize` về $\sigma_0 = 0.07925$.
   - Ở **Low/Mid SNR ($-5\text{ dB} \to 10\text{ dB}$)**: Công suất nhiễu nền lớn hoặc tương đương tiếng nói. FRCRN tách $\hat{\mathbf{s}}$ ra để lại $\hat{\mathbf{n}}$ giữ nguyên cấu trúc phổ của 36 lớp âm thanh môi trường.
   - Ở **High SNR ($15\text{ dB} \to 20\text{ dB}$)**: Công suất tiếng nói mạnh gấp **$31.6\times$ đến $100\times$** công suất nhiễu nền ($10^{20/10} = 100$). Chỉ cần FRCRN ước lượng sai lệch $1\% - 2\%$ biên độ/pha của tiếng nói $\hat{\mathbf{s}}$, phần rò rỉ tiếng nói (speech leakage) và các hố triệt tiêu phổ (spectral masking holes) trong $\hat{\mathbf{n}} = \mathbf{x}_{\text{mix}} - \hat{\mathbf{s}}$ sẽ có năng lượng tương đương hoặc lớn hơn cả tín hiệu nhiễu thực sự! Khi `rms_normalize` phóng đại $\hat{\mathbf{n}}$ ở `20 dB` lên `RMS = 0.07925`, các giả ảnh (artifacts) này cũng bị khuếch đại gấp ~10 lần.
   - Do đó, phân phối đặc trưng bậc cao $P(\mathbf{H}_{\text{deep}} \mid Y, \text{SNR})$ ở `20 dB` khác biệt mạnh so với `-5..10 dB`. Một Classifier Head đơn lẻ bị thống trị bởi $5/6$ lượng mẫu ở SNR thấp/trung bình.

2. **Tại sao KHÔNG cần 2 Backbone 12 tầng riêng biệt?**
   - Trong Vision/Audio Transformer (`ViTSBias`), `PatchEmbed` và các tầng đầu/giữa (Block `0..8`) đóng vai trò bộ lọc cơ sở (edge/harmonic detectors, local onset/offset, formant/bark-band grouping). Các đặc trưng âm học cơ sở này là **bất biến và dùng chung** cho cả Low-SNR lẫn High-SNR.
   - Sự xung đột gradient giữa Low-SNR và High-SNR chỉ xảy ra ở **các tầng ngữ nghĩa trừu tượng cuối cùng (Block `10..11`)** — nơi mô hình quyết định xem một dải formant là "tiếng người thật trong lớp nền" hay "rò rỉ tiếng nói từ FRCRN ở 20 dB" — và tại **Classifier Head**.

---

## 3. Giả Thuyết Nguyên Nhân Gốc & Kiểm Chứng (Ranked Root-Cause Hypotheses)

1. **Giả thuyết H1 (Xung đột biểu diễn tầng sâu & thiên lệch đa số - Độ tin cậy: Cao):**
   - *Quan sát:* Một model CED-Tiny đơn lẻ (`kd-v5`) đạt kết quả rất cao ở `-5..10 dB` nhưng suy giảm mạnh ở `20 dB`; khi fine-tune riêng một `High-Student` cho `20 dB` trong `imp/moe`, độ chính xác tại `20 dB` tăng `+6.88%`.
   - *Cơ chế:* Trong quá trình huấn luyện chung trên 6 mức SNR, $83.3\%$ gradient đến từ $\text{SNR} \le 15\text{ dB}$. Các tầng FFN cuối và Linear Head bị tối ưu lệch về phía tin tưởng hoàn toàn vào phổ năng lượng mạnh, dẫn đến việc nhầm speech leakage ở `20 dB` thành các lớp âm thanh có cấu trúc hài (như *Singing, Female speech, Clarinet, Saxophone*).
   - *Phép thử phân biệt:* Đóng băng Block `0..9` của `low_student`, chỉ tách nhánh FFN ở Block `10..11` và `head` cho chế độ High-SNR — nếu hiệu năng `20 dB` khôi phục tương đương `high_student` độc lập, giả thuyết H1 được xác nhận.

2. **Giả thuyết H2 (Overfitting của chuyên gia High-SNR khi dùng toàn bộ 5.42M tham số - Độ tin cậy: Cao):**
   - *Quan sát:* Tập dữ liệu ở mức `20 dB` chỉ bằng $1/6$ tổng tập train ($\approx 5,040$ mẫu trên 36 lớp, tức chỉ $\sim 140$ mẫu/lớp).
   - *Cơ chế:* Huấn luyện một mạng độc lập `5.42M` tham số chỉ cho miền High-SNR dễ dẫn đến học vẹt (memorization) và lãng phí dung lượng, trong khi nếu chia sẻ 80–95% tham số ở thân mạng và chỉ dành `0.15M – 0.60M` tham số cho nhánh chuyên gia, nhánh High-SNR được hưởng lợi từ hiệu ứng chính quy hóa (regularization) của toàn bộ 30,240 mẫu ở các tầng đầu.

3. **Giả thuyết H3 (Định tuyến cứng tại $\tau = 18\text{ dB}$ gây mất liên tục ở vùng chuyển tiếp $10\text{ dB} - 20\text{ dB}$ và phụ thuộc Oracle SNR - Độ tin cậy: Cao):**
   - *Quan sát:* Trong `imp/moe/README.md`, độ chính xác ở `15 dB` (`57.79%`) là vùng chuyển tiếp giảm mạnh so với `10 dB` (`62.96%`), và `audit/README.md` chỉ rõ `evaluate_pipeline` đang truyền `snr_db` thật (Oracle routing).
   - *Cơ chế:* Sự suy biến của $\hat{\mathbf{n}}$ diễn ra liên tục từ `10 dB` $\to$ `15 dB` $\to$ `20 dB` và phụ thuộc vào từng mẫu cụ thể (sample-dependent SDR của FRCRN), không phải hàm bậc thang cứng tại `18 dB`. Một bộ định tuyến mềm (Soft Gating Router) học từ đặc trưng của $\hat{\mathbf{n}}$ sẽ giải quyết đồng thời cả vùng chuyển tiếp `15 dB` lẫn bài toán không có `snr_db` lúc test.

---

## 4. Khảo Sát Các Bài Báo & Phương Pháp Từ Bên Ngoài (Verified Literature Evidence)

| STT | Công trình & Nguồn xác thực | Vấn đề giải quyết & Cơ chế cốt lõi | Mức độ phù hợp với CED-Tiny (`imp/moe`) |
| :-: | :--- | :--- | :--- |
| **1** | **V-MoE:** *Scaling Vision with Sparse Mixture of Experts* (Riquelme et al., Google Brain, **NeurIPS 2021**, [arXiv:2106.05974](https://arxiv.org/abs/2106.05974)) | Thay vì đặt MoE ở mọi khối ViT, tác giả chứng minh chiến lược **"Last-$K$"** (chỉ thay thế tầng MLP/FFN bằng các Experts ở $K=2$ khối cuối cùng của ViT, giữ nguyên Attention và toàn bộ các khối đầu) đạt hiệu năng tương đương hoặc tốt hơn đặt MoE toàn mạng, đồng thời giảm mạnh chi phí bộ nhớ và ổn định gradient. | **Rất Cao (Trực tiếp):** Trong CED-Tiny (`5.42M`), 12 khối MLP chiếm `3.55M` tham số (`65.5%` toàn mạng). Chỉ tách FFN ở **2 khối cuối (Block 10 & 11)** giúp tái sử dụng 10 khối đầu làm backbone chung và chỉ tăng `~0.59M` tham số cho mỗi chuyên gia tăng thêm. |
| **2** | **Soft MoE:** *From Sparse to Soft Mixtures of Experts* (Puigcerver et al., Google DeepMind, **ICLR 2024**, [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)) | Thay thế định tuyến rời rạc (Hard Top-1 gating gây bất ổn định huấn luyện và rơi token) bằng **tổ hợp tuyến tính mềm khả vi hoàn toàn (Soft Convex Combination)** giữa các chuyên gia trong ViT. Không cần loss cân bằng tải phức tạp khi số chuyên gia nhỏ. | **Rất Cao:** Với số lượng chuyên gia nhỏ ($E=2$ hoặc $E=3$ theo dải SNR), chạy **Soft Gating** ($\sum_{e=1}^E w_e(\mathbf{x}) \text{Expert}_e(\mathbf{h})$) khả vi 100%, loại bỏ bước nhảy gián đoạn ở ngưỡng `18 dB` và cải thiện trực tiếp vùng chuyển tiếp `15 dB`. |
| **3** | **HydraLoRA:** *An Asymmetric LoRA Architecture for Efficient Fine-Tuning* (Tian et al., **NeurIPS 2024 Oral**, [arXiv:2404.19245](https://arxiv.org/abs/2404.19245)) & **MoLE** ([arXiv:2404.13628](https://arxiv.org/abs/2404.13628)) | Khi dữ liệu gồm nhiều miền con (sub-domains) gây xung đột gradient, **HydraLoRA** giữ nguyên trọng số gốc $\mathbf{W}_0$, dùng **1 ma trận giảm chiều dùng chung $\mathbf{A} \in \mathbb{R}^{r \times d}$** và **$E$ ma trận nâng chiều chuyên biệt $\mathbf{B}_e \in \mathbb{R}^{d \times r}$** kết hợp qua Router: $\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \sum_{e=1}^E \omega_e \mathbf{B}_e (\mathbf{A}\mathbf{x})$. | **Rất Cao (Tối ưu tham số nhất):** Giữ nguyên 100% trọng số `CED-Tiny` (`5.42M`), gắn HydraLoRA ($r=16$ hoặc $32$) vào các tầng FFN/Attention. Chỉ tốn thêm **`~0.15M` tham số**, không bao giờ phá hỏng kích hoạt gốc (`std=12.5`) của CED-Tiny. |
| **4** | **DeepSeekMoE / Shared-Routed Experts:** *Towards Ultimate Expert Specialization* (Dai et al., **ACL 2024**, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)) | Tách kiến trúc MoE thành **1 Shared Expert (luôn bật cho mọi mẫu)** để gom tri thức chung + **$E$ Routed Experts** để học phần sai khác chuyên biệt (residual specialization) cho từng miền dữ liệu. | **Cao:** Khắc phục triệt để việc High-SNR Expert bị thiếu dữ liệu (`1/6` tập train): Shared Expert học từ 100% dữ liệu (`-5..20 dB`), còn Low-Expert và High-Expert chỉ học phần hiệu chỉnh đặc thù cho từng dải SNR. |
| **5** | **FiLM / adaLN-Zero Conditioning:** *Scalable Diffusion Models with Transformers (DiT)* (Peebles & Xie, **ICCV 2023**, [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)) & *FiLM for Noisy Speech Recognition* (INTERSPEECH) | Điều biến trực tiếp đặc trưng trung gian của Transformer theo vector ngữ cảnh nhiễu $\mathbf{c}$ (như mức SNR ước lượng) qua phép biến đổi affine sau LayerNorm: $\text{adaLN}(\mathbf{h}, \mathbf{c}) = (1 + \gamma(\mathbf{c})) \odot \text{LN}(\mathbf{h}) + \beta(\mathbf{c})$, với $\gamma, \beta$ khởi tạo bằng $0$. | **Cao (Bổ trợ lý tưởng):** Cho phép các tầng sớm/giữa (Block `0..9`) tự điều chỉnh dải động theo mức độ nhiễu của $\hat{\mathbf{n}}$ với chi phí tham số gần như bằng 0 (`< 0.05M`), trước khi đi vào các Expert ở tầng cuối. |
| **6** | **CED** (*Dinkel et al., ICASSP 2024*, [arXiv:2308.11957](https://arxiv.org/abs/2308.11957)) & **MoEKD** (*Multi-Teacher MoE KD*, [arXiv:2603.13213](https://arxiv.org/abs/2603.13213)) | Chưng cất nhất quán (Consistent Teaching) trên cùng phép biến đổi dữ liệu (Mixup/SpecAugment) kết hợp tổng hợp tri thức từ nhiều giáo viên chuyên biệt (Multi-Teacher Routing Aggregation) vào một mô hình sinh viên duy nhất. | **Rất Cao:** Giải thích vì sao trong `ChampionPipeline`, `Teacher Fold0` mạnh ở Low-SNR còn `Teacher EMA` mạnh ở High-SNR; ta có thể chưng cất trực tiếp tổ hợp hai Teacher này vào 1 Student duy nhất. |

---

## 5. Phân Tích & So Sánh 4 Phương Án Tối Ưu Hóa Các Expert Trong 1 Backbone Duy Nhất

Dưới đây là 4 phương án thiết kế cụ thể để gom `low_student` (`5.42M`) + `high_student` (`5.42M`) thành **1 mô hình CED-Tiny duy nhất**:

### Phương Án 1: `CED-SplitTopK-MoE` (Shared Stem + Last-2 Blocks Split-FFN & Multi-Head) — 🌟 KHUYẾN NGHỊ SỐ 1
* **Cấu trúc:**
  - **Tầng dùng chung (Block 0 $\to$ Block 9 + toàn bộ `BiasAttention` của Block 0 $\to$ 11):** Dùng chung 1 bộ trọng số duy nhất (`4.82M` tham số).
  - **Tầng chuyên gia (Block 10 & Block 11 MLP + Classifier Head):** Thay thế `mlp` ở 2 block cuối và `head` bằng **1 Shared Branch** + **2 Routed Expert Branches (Low-Expert & High-Expert)** (hoặc 2 nhánh Low/High thuần túy được trộn mềm):
    $$\mathbf{h}_{\text{out}}^{(l)} = \mathbf{h}_{\text{attn}}^{(l)} + (1 - \alpha(\mathbf{x})) \cdot \text{FFN}_{\text{low}}\left(\text{LN}(\mathbf{h}_{\text{attn}}^{(l)})\right) + \alpha(\mathbf{x}) \cdot \text{FFN}_{\text{high}}\left(\text{LN}(\mathbf{h}_{\text{attn}}^{(l)})\right), \quad l \in \{10, 11\}$$
    $$\mathbf{z} = (1 - \alpha(\mathbf{x})) \cdot \mathbf{W}_{\text{head, low}} \mathbf{h}_{\text{cls}} + \alpha(\mathbf{x}) \cdot \mathbf{W}_{\text{head, high}} \mathbf{h}_{\text{cls}}$$
  - **Khởi tạo thông minh (Warm-Start):** Cả `FFN_low` và `FFN_high` đều được copy trọng số từ `mispeech/ced-tiny` (hoặc từ checkpoint `kd-v5` đã hội tụ), chỉ khác nhau ở quá trình nhận gradient có trọng số theo $(1 - \alpha(\mathbf{x}))$ và $\alpha(\mathbf{x})$!
* **Số tham số:** `5.42M` (gốc) + `2 × 0.295M` (2 FFN ở Block 10, 11) + `0.007M` (1 Head phụ) + `0.005M` (Router) = **`6.02M` tham số** (Giảm **44.5%** so với `10.84M` của 2 model rời).
* **Chi phí tính toán (FLOPs):** `PatchEmbed` và 10 Block đầu + Attention của 2 Block cuối chỉ chạy **1 lần duy nhất** (tiết kiệm **~88% FLOPs** so với chạy 2 model CED-Tiny).

### Phương Án 2: `CED-HydraLoRA-MoE` (Shared Frozen/Active Backbone + Asymmetric LoRA Experts + Dual Head)
* **Cấu trúc:**
  - Giữ nguyên **100% cấu trúc `ViTSBias` (`5.42M` tham số)** không nhân bản bất kỳ FFN nào.
  - Tại các tầng `mlp.fc1` và `mlp.fc2` từ **Block 6 đến Block 11** (6 block nửa sau), tiêm module **HydraLoRA** (rank $r = 16$ hoặc $32$):
    $$\text{Linear}_{\text{Hydra}}(\mathbf{u}) = \mathbf{W}_0 \mathbf{u} + \frac{s}{r} \left( (1 - \alpha(\mathbf{x})) \mathbf{B}_{\text{low}} + \alpha(\mathbf{x}) \mathbf{B}_{\text{high}} \right) (\mathbf{A} \mathbf{u})$$
    Trong đó $\mathbf{W}_0$ là trọng số dùng chung, $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ là phép chiếu giảm chiều dùng chung, và $\mathbf{B}_{\text{low}}, \mathbf{B}_{\text{high}} \in \mathbb{R}^{d_{\text{out}} \times r}$ là 2 chuyên gia hạng thấp (khởi tạo bằng `0` để bảo toàn tuyệt đối phương sai kích hoạt `std = 12.5` của CED-Tiny ở epoch đầu!).
  - Kết hợp với 2 Classifier Heads (`Linear(192, 36)`) ở đầu ra.
* **Số tham số:** `5.42M` + `0.15M` (HydraLoRA $r=16$ trên 6 blocks) + `0.007M` (Head) = **`5.58M` tham số** (Tăng chưa tới **3%** so với 1 model đơn!).
* **Ưu điểm đặc biệt:** Chống overfitting cực mạnh cho chuyên gia High-SNR vì số tham số riêng của chuyên gia High-SNR chỉ khoảng `~75K` tham số (so với `5,040` mẫu ở `20 dB`).

### Phương Án 3: `CED-adaLN-MoE` (SNR-Conditioned Adaptive LayerNorm + Multi-Expert Head)
* **Cấu trúc:**
  - Một bộ trích xuất đặc trưng nhiễu nhẹ $\mathbf{c} = \text{RegimeEncoder}(\hat{\mathbf{n}}) \in \mathbb{R}^{64}$ dự đoán hệ số điều biến $(\gamma_l, \beta_l)$ cho tất cả `LayerNorm` của 12 khối Transformer:
    $$\text{LN}_{\text{ada}}(\mathbf{h}; \mathbf{c}) = (1 + \gamma_l(\mathbf{c})) \odot \text{LN}(\mathbf{h}) + \beta_l(\mathbf{c})$$
  - Ở đầu ra sau tầng Norm cuối, đặt $E=3$ Expert Heads (Low: `[-5, 0] dB`, Mid: `[5, 10, 15] dB`, High: `[20] dB`) trộn mềm qua xác suất định tuyến $\mathbf{g}(\mathbf{c}) \in \Delta^2$.
* **Số tham số:** **`5.49M` tham số** (`+0.07M`).

### Phương Án 4: Giải Pháp Thiết Kế Bộ Định Tuyến Nội Tại (`Self-Contained Spectral-Token Router`) — Áp dụng cho cả Phương án 1, 2, 3
Một điểm yếu chí mạng của `imp/moe` hiện tại là `EnergySNRRouter` cần cả `mixture` ($\mathbf{x}_{\text{mix}}$) và `noise` ($\hat{\mathbf{n}}$) để tính $\text{SNR} = 10\log_{10}(P_{\text{clean}}/P_{\text{noise}})$, nhưng khi chạy từ cache `.npy` chỉ có $\hat{\mathbf{n}}$ nên pipeline phải truyền `snr_db` thật (Oracle). Để **1 model tự vận hành hoàn toàn chỉ từ đầu vào $\hat{\mathbf{n}}$**:
* **Đầu vào của Router nội tại $\phi(\hat{\mathbf{n}}, \mathbf{H}^{(l)})$:**
  1. **Đặc trưng vật lý trước chuẩn hóa RMS (1 chiều):** $\log(\text{RMS}(\hat{\mathbf{n}}_{\text{raw}}) + \epsilon)$ (nếu cache lưu $\hat{\mathbf{n}}$ trước khi chuẩn hóa RMS, biên độ gốc của $\hat{\mathbf{n}}$ phản ánh trực tiếp mức năng lượng nhiễu!).
  2. **Đặc trưng hình thái phổ Log-Mel (5 chiều, bất biến với RMS scaling):**
     - *Spectral Flatness (Độ phẳng phổ Wiener entropy):* Phân biệt nhiễu nền rộng với hài âm tiếng nói rò rỉ ở `20 dB`.
     - *Sub-band Energy Ratio (4 dải tần):* Tỷ lệ năng lượng dải tần tiếng nói (`300 Hz – 3400 Hz`) so với dải tần thấp (`<300 Hz`) và cao (`>3400 Hz`).
     - *Temporal Frame Energy Kurtosis & Crest Factor:* Ở `20 dB`, phần rò rỉ âm tiết tiếng nói từ FRCRN tạo ra các đỉnh nhọn theo khung thời gian (high kurtosis), khác hẳn nhiễu nền ở `-5..5 dB`.
  3. **Đặc trưng ngữ nghĩa từ tầng giữa của ViT (192 chiều, stop-gradient hoặc chiếu xuống 16 chiều):** Vector `cls_token` hoặc trung bình token tại Block 9.
* **Huấn luyện Router (Supervised + End-to-End Soft Gating):**
  - Trong tập train, ta biết chính xác `snr_db` $\in \{-5, 0, 5, 10, 15, 20\}$. Ta định nghĩa nhãn định tuyến mềm mục tiêu:
    $$y_{\text{route}} = \sigma\left(\frac{\text{SNR}_{\text{dB}} - 15.0}{\tau_r}\right) \in (0, 1) \quad (\text{ví dụ } \tau_r = 2.5 \implies -5\text{ dB} \approx 0.00, \; 15\text{ dB} = 0.50, \; 20\text{ dB} = 0.88)$$
  - Hàm mất mát huấn luyện Router: $\mathcal{L}_{\text{route}} = \text{MSE}(\alpha(\mathbf{x}), y_{\text{route}})$ hoặc $\text{BCE}(\alpha(\mathbf{x}), y_{\text{route}})$ cộng vào tổng loss với hệ số $\lambda_{\text{route}} = 0.2$.
  - Nhờ nhãn mềm liên tục này, tại **`15 dB` (vùng chuyển tiếp)**, $\alpha(\mathbf{x}) \approx 0.5$ sẽ tự động pha trộn mượt mà cả Low-Expert và High-Expert thay vì cắt cụt ở `18 dB`!

### Bảng So Sánh Trade-off Giữa Các Phương Án

| Tiêu chí | Hiện tại: `ChampionPipeline` (`imp/moe`) | Phương án 1: `CED-SplitTopK-MoE` (Last-2 FFN + Dual Head) | Phương án 2: `CED-HydraLoRA-MoE` (LoRA Experts + Dual Head) | Phương án 3: `CED-adaLN-MoE` (FiLM/adaLN + Multi-Head) |
| :--- | :---: | :---: | :---: | :---: |
| **Số lượng Backbone** | 2モデル rời (`S1` + `S2`) | **1 Backbone duy nhất** | **1 Backbone duy nhất** | **1 Backbone duy nhất** |
| **Tổng tham số Student** | `10.84M` (`2 × 5.42M`) | **`6.02M`** (`+11%` so với đơn) | **`5.58M`** (`+2.9%` so với đơn) | **`5.49M`** (`+1.3%` so với đơn) |
| **FLOPs suy luận (Student)** | `2.00×` (nếu chạy cả 2) | **`~1.11×`** (chỉ tách 2 block cuối) | **`~1.03×`** | **`~1.01×`** |
| **Dung lượng chuyên biệt hóa** | Cao nhưng thừa ở tầng nông | **Rất Cao** (`0.60M` ở đúng tầng ngữ nghĩa sâu) | **Cao & Chống overfit tốt** (`0.16M` hạng thấp) | Trung bình (chỉ điều biến tuyến tính theo kênh) |
| **Bảo toàn kích hoạt CED (`std=12.5`)** | Tốt | **Tuyệt đối** (khởi tạo 2 nhánh từ chính trọng số CED) | **Tuyệt đối** ($\mathbf{B}_k$ khởi tạo bằng `0`) | Tốt ($\gamma, \beta$ khởi tạo bằng `0`) |
| **Xử lý vùng chuyển tiếp `15 dB`** | Kém (`57.79%`, cắt cứng ở `18 dB`) | **Tối ưu** (Soft Gating nội suy mượt ở `15 dB`) | **Tối ưu** (Soft Gating nội suy mượt) | Tốt |
| **Độ khó triển khai** | Đã có sẵn | **Thấp – Trung bình** (~150 dòng code trong `vit_bias.py`) | **Trung bình** (cần viết wrapper `HydraLinear`) | **Thấp** |

---

## 6. Chiến Lược Huấn Luyện & Chưng Cất Đa Thầy (Dual-Teacher Regime Distillation)

Để **1 model Student duy nhất** hấp thụ được tinh hoa của cả `Teacher Fold0` (mạnh ở `-5..15 dB`) và `Teacher EMA` (mạnh ở `20 dB`) mà không cần giữ 2 Teacher nặng `180M` tham số lúc triển khai độc lập (hoặc vẫn kết hợp được nếu chạy ensemble):

1. **Hàm Mục Tiêu Tổ Hợp Theo Chế Độ Nhiễu (Regime-Weighted Teacher Target):**
   Với mỗi mẫu huấn luyện có mức $\text{SNR}_i$ và hệ số trộn mềm mục tiêu $w_i = \sigma((\text{SNR}_i - 15.0) / 2.5) \in [0, 1]$:
   - Phân phối mục tiêu cho **nhánh Low-Expert ($\mathbf{z}_{\text{low}}$)**:
     $$\mathbf{P}_{T, \text{low}}^{(T)} = 0.65 \cdot \text{Softmax}(\mathbf{z}_{T, \text{fold0}} / T) + 0.35 \cdot \text{Softmax}(\mathbf{z}_{T, \text{ema}} / T)$$
   - Phân phối mục tiêu cho **nhánh High-Expert ($\mathbf{z}_{\text{high}}$)**:
     $$\mathbf{P}_{T, \text{high}}^{(T)} = 0.30 \cdot \text{Softmax}(\mathbf{z}_{T, \text{fold0}} / T) + 0.70 \cdot \text{Softmax}(\mathbf{z}_{T, \text{ema}} / T)$$
   - Phân phối mục tiêu cho **đầu ra hợp nhất của Student ($\mathbf{z}_{\text{fused}} = (1 - \hat{\alpha}_i)\mathbf{z}_{\text{low}} + \hat{\alpha}_i \mathbf{z}_{\text{high}}$)**:
     $$\mathbf{P}_{T, \text{fused}}^{(T)} = (1 - w_i) \cdot \mathbf{P}_{T, \text{low}}^{(T)} + w_i \cdot \mathbf{P}_{T, \text{high}}^{(T)}$$

2. **Loss Chưng Cất Chuyên Gia Có Trọng Số Mẫu (Sample-Weighted Expert Specialization Loss):**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\mathbf{z}_{\text{fused}}, \tilde{\mathbf{y}}) + \lambda_{\text{kd}} \mathcal{L}_{\text{KL}}(\mathbf{z}_{\text{fused}}, \mathbf{P}_{T, \text{fused}}^{(T)}) + \lambda_{\text{spec}} \Big[ (1 - w_i) \mathcal{L}_{\text{KL}}(\mathbf{z}_{\text{low}}, \mathbf{P}_{T, \text{low}}^{(T)}) + w_i \mathcal{L}_{\text{KL}}(\mathbf{z}_{\text{high}}, \mathbf{P}_{T, \text{high}}^{(T)}) \Big] + \lambda_{\text{route}} \text{MSE}(\hat{\alpha}_i, w_i)$$
   - **Ý nghĩa:** Số hạng $\lambda_{\text{spec}}$ buộc nhánh `Low-Expert` tập trung tối đa vào các mẫu SNR thấp/trung bình và nhánh `High-Expert` tập trung tối đa vào các mẫu SNR cao (`15–20 dB`), trong khi cả hai vẫn cùng tinh chỉnh thân mạng chung (Block `0..9`).
   - **Xử lý Mixup (`mixup_alpha = 0.4, prob = 0.8`):** Khi trộn 2 mẫu $i$ và $j$ theo tỷ lệ $\lambda_{\text{mix}}$:
     $$w_{\text{mix}} = \lambda_{\text{mix}} w_i + (1 - \lambda_{\text{mix}}) w_j$$
     Điều này tạo ra vô số mức SNR trung gian liên tục trong lúc train, giúp Router và 2 nhánh Expert học được bề mặt nội suy trơn tru!

---

## 7. Kế Hoạch Thực Nghiệm Ưu Tiên (Prioritized Validation Plan)

### Thí nghiệm 1 (Ưu tiên số 1 - Chi phí thấp, Lợi ích thông tin cao nhất): `CED-SplitTop2-MoE` (Warm-start từ Checkpoint `kd-v5`)
* **Mục tiêu:** Kiểm chứng việc gộp 2 model thành 1 backbone duy nhất với Split-FFN ở 2 block cuối (Block 10, 11) + Dual Head + Internal Spectral-Token Router.
* **Cách khởi tạo nhanh (Fast Warm-Start trong 10–12 epochs):**
  - Thay vì train từ đầu 30 epochs, nạp trực tiếp checkpoint `low_student` (`kd-v5` best checkpoint) vào toàn bộ thân mạng Block `0..9` và nhánh `Low-Expert` (Block `10..11` FFN_low + Head_low).
  - Khởi tạo nhánh `High-Expert` (Block `10..11` FFN_high + Head_high) từ `high_student` (nếu có sẵn) hoặc nhân bản từ `low_student`.
  - Huấn luyện tinh chỉnh (fine-tune) với `lr = 5e-5` cho thân mạng dùng chung và `lr = 1e-4` cho Router + 2 nhánh Expert trong `12 epochs` với Dual-Teacher KD.
* **Tiêu chí chấp nhận (Acceptance Criteria trên tập Validation Clean 3,064 mẫu — không dùng Oracle SNR):**
  1. **Số lượng mô hình Student:** Duy nhất **1 model** (`~6.02M` tham số, giảm từ `10.84M`).
  2. **Standalone Student (không kèm Teacher lúc suy luận):** Val Macro-Acc $\ge 0.632$ và Val mAP $\ge 0.703$ (vượt qua `kd-v5` đơn lẻ `0.629 / 0.701`), đặc biệt độ chính xác tại `20 dB` tăng tối thiểu **`+3.5% – +5.0%`** so với 1 model đơn.
  3. **Khi đặt trong Champion Pipeline (thay thế `S1 + S2` bằng `1× CED-SingleMoE`):** Đạt Top-1 Accuracy $\ge 63.50\%$ và Macro mAP $\ge 70.0\%$ **mà không cần truyền `snr_db` thật (Non-Oracle Routing)**.

### Thí nghiệm 2 (Phương án dự phòng / Siêu nén): `CED-HydraLoRA-MoE` ($r=16$, Block 6..11)
* **Khi nào chạy:** Nếu Thí nghiệm 1 cho thấy nhánh `High-Expert` ở Block 10–11 (`0.60M` tham số) có dấu hiệu overfit trên mức `20 dB`, chuyển sang gắn **HydraLoRA ($r=16$)** vào 6 block cuối (chỉ `0.16M` tham số phụ).

### Thí nghiệm 3 (Nâng cao): Mở rộng lên 3 Chuyên Gia (`Low: [-5, 0] dB`, `Mid: [5, 10, 15] dB`, `High: [20] dB`) + Shared Expert
* **Cơ chế:** Theo đúng cấu trúc **DeepSeekMoE**, giữ 1 FFN cố định (Shared Expert) và 3 FFN hạng nhẹ (Bottleneck dimension `384` thay vì `768` để tổng tham số của 3專家 chỉ bằng 1.5× FFN gốc) tương ứng với 3 chế độ âm học: Nhiễu thống trị (`-5, 0 dB`), Cân bằng (`5, 10, 15 dB`), và Giả ảnh tách nguồn (`20 dB`).

---

## 8. Tài Liệu Tham Khảo (References)

1. **V-MoE:** C. Riquelme et al., *"Scaling Vision with Sparse Mixture of Experts,"* in Proc. NeurIPS, 2021. [arXiv:2106.05974](https://arxiv.org/abs/2106.05974) (Mục 3.2 & Bảng 2: Phân tích đặt MoE ở *Last-2 layers* so với *Every-2 layers*).
2. **Soft MoE:** J. Puigcerver et al., *"From Sparse to Soft Mixtures of Experts,"* in Proc. ICLR, 2024. [arXiv:2308.00951](https://arxiv.org/abs/2308.00951) (Cơ chế Soft Routing khả vi cho Vision Transformers).
3. **HydraLoRA:** C. Tian et al., *"HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning,"* in Proc. NeurIPS (Oral), 2024. [arXiv:2404.19245](https://arxiv.org/abs/2404.19245) (Cấu trúc 1 ma trận $A$ dùng chung + $K$ ma trận $B_k$ chuyên gia).
4. **MoLE:** X. Wu et al., *"Mixture of LoRA Experts,"* in Proc. ICLR, 2024. [arXiv:2404.13628](https://arxiv.org/abs/2404.13628).
5. **DeepSeekMoE:** D. Dai et al., *"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models,"* in Proc. ACL, 2024. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) (Tách Shared Expert và Routed Experts).
6. **CED:** H. Dinkel et al., *"CED: Consistent Ensemble Distillation for Audio Tagging,"* in Proc. ICASSP, 2024. [arXiv:2308.11957](https://arxiv.org/abs/2308.11957).
7. **BEATs:** S. Chen et al., *"BEATs: Audio Pre-Training with Acoustic Tokenizers,"* in Proc. ICML, 2023. [arXiv:2212.09058](https://arxiv.org/abs/2212.09058).
8. **DiT (adaLN-Zero):** W. Peebles and S. Xie, *"Scalable Diffusion Models with Transformers,"* in Proc. ICCV, 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).
9. **MoEKD:** M. A. Awal et al., *"MoEKD: Mixture-of-Experts Knowledge Distillation for Robust and High-Performing Compressed Models,"* 2026. [arXiv:2603.13213](https://arxiv.org/abs/2603.13213).
