# Tiếng Anh học thuật

## Mục lục
1. Đọc nhanh bài báo
2. Cấu trúc câu và từ nối
3. Viết thận trọng (hedging)
4. Mẫu câu theo mục
5. Lỗi thường gặp của người nói tiếng Việt
6. Từ vựng ML và âm thanh hay gặp

## 1. Đọc nhanh bài báo

Không đọc từ đầu đến cuối ngay. Thứ tự:

1. **Title, abstract, figure 1:** bài làm gì, so với cái gì.
2. **Introduction, câu cuối đoạn cuối:** thường là danh sách "Our contributions are...".
3. **Method:** tìm các phương trình định nghĩa đóng góp.
4. **Experiments:** bảng kết quả và ablation ("Ablation study", "Analysis").
5. **Limitations, Conclusion.**

Khi gặp câu dài, tìm **động từ chính** và **chủ ngữ**, rồi mới xét mệnh đề phụ. Từ báo hiệu ý chính: *we propose, we show, in contrast, however, notably, surprisingly, in particular*.

## 2. Cấu trúc câu và từ nối

| Chức năng | Mẫu |
|---|---|
| Nêu vấn đề | *Existing methods struggle to... because...* |
| Nêu đóng góp | *We propose X, which... Our key insight is that...* |
| Đối lập | *In contrast to..., ... / Unlike..., ...* |
| Nguyên nhân | *This is because... / This is due to...* |
| Hệ quả | *As a result, ... / Consequently, ...* |
| Bổ sung | *Moreover, ... / In addition, ...* |
| Ví dụ | *For instance, ... / For example, ...* |
| Giới hạn | *A limitation of our approach is that...* |

Mỗi đoạn một ý; câu đầu là câu chủ đề.

## 3. Viết thận trọng (hedging)

Khoa học không khẳng định quá bằng chứng. Thang mức độ:

| Mạnh | Vừa | Yếu |
|---|---|---|
| *proves, demonstrates* | *shows, indicates* | *suggests, is consistent with* |
| *always, never* | *typically, generally* | *may, might, could* |
| *is caused by* | *is likely due to* | *may be attributed to* |

Ví dụ:

- Quá mạnh: *The router fails because the teacher is wrong.*
- Đúng mức: *The low router accuracy is consistent with unreliable teacher outputs at high SNR, although we have not tested this directly.*

Dùng "may" khi là giả thuyết, "shows" khi có số liệu của chính bạn, "has been shown" khi dẫn nguồn.

## 4. Mẫu câu theo mục

**Abstract**
- *Classifying background noise under speech is challenging at high SNR, where the noise is largely masked.*
- *We distill a frozen teacher into a compact student with a mixture-of-experts head.*
- *Our student reaches X% of the teacher's accuracy with Y% of its parameters.*

**Method**
- *Given a mixture x, we estimate the noise component n̂ using...*
- *We use a temperature T = 2 to soften the teacher distribution.*

**Results**
- *Table 2 reports macro accuracy across six SNR levels.*
- *Performance degrades as SNR increases, in line with the masking effect.*

**Limitations**
- *Our results are based on a single seed.*
- *We did not evaluate on out-of-domain recordings.*

## 5. Lỗi thường gặp của người nói tiếng Việt

| Lỗi | Sửa | Ghi chú |
|---|---|---|
| Thiếu mạo từ: *We propose method.* | *We propose a method.* / *the proposed method* | Danh từ đếm được số ít cần *a/the* |
| Số nhiều: *many noise* | *many noises / much noise* | *noise* thường không đếm được; *noises* khi nói các loại |
| Thì: *The model is trained yesterday.* | *The model was trained yesterday.* | Kết quả đã làm dùng quá khứ; chân lý hiện hành dùng hiện tại |
| *Result show...* | *Results show...* | Hoà hợp chủ ngữ và động từ |
| *In the other hand* | *On the other hand* | Cặp *On the one hand / On the other hand* |
| *More better* | *better* | So sánh hơn không lặp |
| *Based on the above, we can conclude that...* nhiều lần | Dùng thay đổi: *Thus, ...* | Tránh lặp |
| *We use X to do Y* (mơ hồ) | *We use X to estimate Y* | Chọn động từ cụ thể |

## 6. Từ vựng ML và âm thanh hay gặp

| Tiếng Anh | Nghĩa |
|---|---|
| ablation study | khảo sát bỏ từng thành phần |
| baseline | mô hình đối chứng |
| distillation / teacher / student | chưng cất / mô hình thầy / trò |
| signal-to-noise ratio (SNR) | tỉ số tín hiệu trên nhiễu |
| mixture / mixed signal | tín hiệu trộn |
| log-mel spectrogram | phổ mel dạng log |
| router / gating network | bộ định tuyến / mạng cổng |
| out-of-fold | dự đoán trên fold bị giữ lại |
| held-out | tập giữ riêng, chưa thấy khi huấn luyện |
| overfitting / generalise | quá khớp / tổng quát hoá |
| robust to | bền vững với |
| outperform | vượt trội hơn |
| comparable to | tương đương với |
| to the best of our knowledge | theo hiểu biết của chúng tôi |
