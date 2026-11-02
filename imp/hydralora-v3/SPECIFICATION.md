# Kỹ thuật Đặc tả Kiến trúc và Hệ thống (Technical Specification)
## CED-HydraLoRA-v3: High-Accuracy Single-Backbone MoE (~5.87M Params)

---

### 1. Giới thiệu và Bối cảnh Bài toán
Trong hệ thống phân loại âm thanh đa điều kiện (Mixture-36), thách thức lớn nhất là sự mất cân bằng hiệu năng giữa các dải tỷ số tín hiệu trên nhiễu (SNR):
- Tại **Low-SNR (-5dB, 0dB)**: Nhiễu môi trường chiếm ưu thế; các mô hình tiền huấn luyện âm học (như BEATs và CED-Tiny) đạt độ chính xác tương đối cao nhờ cấu trúc phổ nhiễu nhất quán.
- Tại **High-SNR (15dB, 20dB)**: Thuật toán tiền xử lý khử nhiễu (FRCRN speech separation) tạo ra hiện tượng **rò rỉ sóng hài tiếng nói (speech harmonics leakage)**. Các xung đột biến này phá vỡ biểu diễn âm học môi trường, khiến các mô hình giáo viên (BEATs Fold0 và EMA) bị nhầm lẫn và hallucinate đến **42.83% số mẫu tại 20dB**.
- Trong các nghiên cứu trước, việc chưng cất tri thức tiêu chuẩn (Standard KD) đã vô tình ép mô hình học sinh (Student) phải học cả những nhãn sai của Teacher, làm giảm hiệu năng ở 20dB từ 46.37% xuống 44.20%. Để đạt mốc kỷ lục 63.94%, hệ thống cũ (Champion Pipeline v2) buộc phải giữ cả 2 mô hình BEATs ở thời điểm suy luận (Inference), dẫn đến tổng số tham số lên tới **186.4M - 187.3M**.

**Mục tiêu của CED-HydraLoRA-v3**:
Đạt độ chính xác $\ge 62\% \to 63\%$ trong chế độ **suy luận độc lập 1-pass hoàn toàn (Standalone Inference)** mà **CHỈ SỬ DỤNG ~5.87M tham số**, triệt để không phụ thuộc vào BEATs hay bất kỳ Teacher nào lúc chạy thực tế.

---

### 2. Bảng Phân bổ Ngân sách Tham số (Parameter Budget)

| Thành phần Module | Chi tiết Kiến trúc | Số lượng Tham số | Tỷ lệ (%) |
| :--- | :--- | :---: | :---: |
| **CED-Tiny Base ViT** | 12 Blocks Pre-LN ViT ($D=192$, 3 Heads), AudioSet Pretrained | 5,423,431 | 92.4% |
| **2D Relative Position Bias** | Bảng nhúng vị trí tương đối thời gian - tần số ($\Delta f \in [-3, 3], \Delta t \in [-23, 23]$) | (Bao gồm trong Base) | - |
| **Asymmetric HydraLoRA** | Blocks 4–11 (8 Blocks): $r_A=24, r_{\text{low}}=8, r_{\text{high}}=24$ trên `mlp.fc1` và `mlp.fc2` | 438,272 | 7.4% |
| **Spectral-Token Router** | 6-dim Acoustic physical features + Mid-layer CLS projection (dim 192 -> 16 -> 32 -> 1) | 3,425 | 0.06% |
| **Dual Regime Heads** | `head_low` (Linear 192 -> 36) + `head_high` (Linear 192 -> 36) | 13,896 | 0.24% |
| **TỔNG CỘNG** | **CED-HydraLoRA-v3 (Ours)** | **5,869,191 (~5.87M)** | **100.0%** |

> [!IMPORTANT]
> Toàn bộ kiến trúc chỉ có **5.869M tham số**, hoàn toàn nằm trong giới hạn $\le 6.0\text{M}$ tham số và nhẹ hơn $31.8\times$ so với Champion Pipeline v2 (186.4M)!

---

### 3. Đột phá Thuật toán và Cơ sở Lý thuyết

#### 3.1. Phễu Lọc Tri thức Đáng Tin Cậy (Sample-Level Reliability Gating)
Để ngăn chặn việc Teacher đầu độc Student tại High-SNR:
1. Với mỗi mẫu âm thanh $i$, kiểm tra tính đúng đắn của dự đoán giáo viên so với Ground Truth $y_i$:
   - $\hat{y}_i^{0} = \arg\max(z_i^{T0})$
   - $\hat{y}_i^{\text{EMA}} = \arg\max(z_i^{T\text{EMA}})$
2. Bốn phân nhánh xử lý:
   - **Trường hợp 1 (Cả 2 giáo viên đều SAI - 42.8% tại 20dB)**:
     $$\lambda_{KD}^{(i)} = 0.0, \quad L_i = L_{CE}(z_i^S, y_i^{\text{smooth}})$$
     Triệt để không cho gradient lỗi của giáo viên tác động vào mô hình học sinh.
   - **Trường hợp 2 (Chỉ Fold0 ĐÚNG)**: Chưng cất $100\%$ từ Fold0.
   - **Trường hợp 3 (Chỉ EMA ĐÚNG)**: Chưng cất $100\%$ từ EMA.
   - **Trường hợp 4 (Cả 2 giáo viên đều ĐÚNG)**: Chưng cất từ phân phối kết hợp mượt theo SNR:
     $$p_i^{\text{target}} = (1 - \alpha_{\text{blend}}) \sigma(z_i^{T0}/T) + \alpha_{\text{blend}} \sigma(z_i^{T\text{EMA}}/T)$$

#### 3.2. Asymmetric HydraLoRA (NeurIPS 2024 Oral)
Tận dụng tính chất bất đối xứng của bài toán:
- Tại Low-SNR: Miền nhiễu có tính phân tán đồng nhất $\implies r_{\text{low}} = 8$ đủ để nắm bắt không gian biến thiên.
- Tại High-SNR: Hiện tượng rò rỉ âm học và sóng hài tiếng nói phức tạp $\implies r_{\text{high}} = 24$ (tăng gấp 3 lần dung lượng biểu diễn mà không làm tăng tổng số tham số nhờ dùng chung ma trận chiếu xuống $A \in \mathbb{R}^{24 \times D}$).
- Công thức chuyển tiếp:
  $$y = W_0 x + (1 - \alpha) \cdot \frac{16}{8} B_{\text{low}} (Ax)_{:8} + \alpha \cdot \frac{32}{24} B_{\text{high}} (Ax)$$

#### 3.3. Huấn luyện Tự Nhất Quán Thời Gian (Temporal Self-Consistency Regularization)
Baking lợi thế của Test-Time Augmentation (TTA) trực tiếp vào trọng số mạng trong 1-pass:
$$L_{\text{cons}} = D_{KL}\left(\sigma\left(\frac{z(x')}{T}\right) \parallel \sigma\left(\frac{z(x)}{T}\right)\right) \cdot T^2$$
với $x' = \text{shift}(x, \Delta t), \Delta t \in \{-50\text{ms}, +50\text{ms}\}$.
Giúp mô hình đạt độ chính xác tương đương TTA ngay trong một lượt suy luận duy nhất (Single Forward Pass).
