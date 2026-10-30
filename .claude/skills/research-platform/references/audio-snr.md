# Kiến thức miền âm thanh: tín hiệu, mel, nhiễu và SNR

## Mục lục
1. Từ sóng âm đến log-mel
2. SNR: định nghĩa và cách trộn
3. Vì sao SNR cao khó phân loại nhiễu
4. Nhóm SNR từ thấp đến cao
5. Các loại nhiễu
6. Kiểm tra mix
7. Bẫy thường gặp

## 1. Từ sóng âm đến log-mel

Tham số của repo (`doc/moe.md` §2.4, `src/model/kdmoe/frontend.py`):

| Bước | Giá trị | Ý nghĩa |
|---|---|---|
| Lấy mẫu | 16 kHz | Nyquist 8 kHz, đủ cho tiếng nói và phần lớn nhiễu cần phân loại |
| Cửa sổ | 400 mẫu (25 ms), n_fft 512 | độ phân giải tần số 16000/512 = 31,25 Hz mỗi bin |
| Bước | 160 mẫu (10 ms) | 4 s → 398 frame |
| Mel | 128 dải, 20–8000 Hz | `m = 2595·log10(1 + f/700)`, dải thấp mịn, dải cao thô |
| Log | `log(x + 1e-6)` trên công suất mel | nén dải động theo cảm nhận độ to |
| Chuẩn hoá | theo clip: `(x − mean)/(std + 1e-5)` | thống kê của **mixture**, áp cho cả nhiễu |
| Cắt | 384 frame | khớp patch 16, lưới 24 theo thời gian |

Đánh đổi cửa sổ: dài thì phân giải tần số tốt, nhoè theo thời gian; ngắn thì ngược lại. 25 ms là điểm quen thuộc cho tiếng nói.

## 2. SNR: định nghĩa và cách trộn

`SNR = 10·log10( P_s / P_n )`, `P = mean(x²)` (công suất RMS trên cả 4 s).

| SNR | P_n so với P_s | Biên độ nhiễu so với tiếng nói |
|---:|---|---|
| −5 dB | gấp 3,16 lần | 1,78 lần |
| 0 dB | bằng nhau | bằng nhau |
| 10 dB | 1/10 | ≈ 0,32 |
| 20 dB | 1/100 | 0,1 |

Quy tắc nhanh: +3 dB ≈ gấp đôi công suất; +10 dB = gấp 10 lần công suất; +20 dB = gấp 10 lần biên độ.

Cách trộn trong repo (`scripts/load_dataset.py::mix_at_snr`): nhân nhiễu với `g = sqrt( P_s / (P_n · 10^(SNR/10)) )`, cộng vào tiếng nói; nếu đỉnh vượt 1,0 thì chia **cả mixture lẫn nhiễu** cho cùng một hệ số để SNR không đổi.

## 3. Vì sao SNR cao khó phân loại nhiễu

1. **Năng lượng:** ở 20 dB nhiễu chỉ chiếm 1% công suất; phần lớn thông tin của mixture là tiếng nói.
2. **Log-mel gần như lấy max:** với hai nguồn không tương quan, công suất cộng, và `log(a + b) ≈ max(log a, log b)` khi a, b lệch nhau. Nhiễu chỉ hiện ở ô thời gian–tần số mà tiếng nói yếu (khoảng lặng, dải tần cao, âm vô thanh).
3. **Bộ tách bị đánh giá khắt khe hơn:** SI-SDR của nhiễu ước lượng rơi từ 13,5 dB (−5 dB) xuống 0,86 dB (20 dB) ở FRCRN fine-tune trong repo.
4. **Hệ quả đo được:** teacher top1 58,7% ở −5 dB xuống 44,3% ở 20 dB; student S 35,4% ở 0 dB xuống 24,1% ở 20 dB.

Nghĩa là "accuracy giảm khi SNR tăng" là hành vi đúng của bài toán *phân loại nhiễu*, ngược với nhận dạng giọng nói (nơi SNR cao dễ hơn).

## 4. Nhóm SNR từ thấp đến cao

Repo dùng ba nhóm cho kịch bản `<nhom>_snr_<mask|nomask>`:

| Nhóm | SNR (dB) | Đặc điểm |
|---|---|---|
| low | −5, 0 | nhiễu lấn hoặc ngang tiếng nói; phân loại dựa nhiều vào nhiễu, dễ nhất cho phân loại nhiễu |
| mid | 5, 10 | tiếng nói trội; nhiễu lộ ở khoảng lặng |
| high | 15, 20 | nhiễu bị che gần hết; cần khử tiếng nói hoặc đặc trưng rất nhạy |

Khi báo kết quả, luôn tách theo SNR (mỗi mốc) và ghi trung bình macro qua mốc; số tổng gộp che mất vùng yếu.

## 5. Các loại nhiễu

Hai trục hữu ích khi giải thích lỗi:

- **Dừng và không dừng:** nhiễu dừng (quạt, động cơ) có phổ gần như không đổi theo thời gian, dễ ước lượng từ khoảng lặng; không dừng (tiếng gõ, còi, tiếng người) đổi theo thời gian, khó tách khỏi tiếng nói.
- **Chồng phổ với tiếng nói:** nhiễu cùng dải 100–4000 Hz (tiếng người, nhạc cụ) khó tách hơn nhiễu ở dải khác; lớp "tiếng người, cơ thể" nằm trong nhóm này.

Nhóm expert E0–E3 của repo theo AudioSet MID (`scripts/build_group_map.py`), không theo hai trục trên; đừng nhầm hai kiểu phân nhóm.

## 6. Kiểm tra mix

Đoạn kiểm nhỏ để xác nhận SNR của mix đã sinh (không phụ thuộc framework):

```python
import numpy as np

def snr_db(speech, noise):
    return 10 * np.log10((np.mean(speech**2) + 1e-12) / (np.mean(noise**2) + 1e-12))

# speech = mixture - noise nếu manifest chỉ lưu mixture và noise
assert abs(snr_db(mixture - noise, noise) - target_snr_db) < 0.1
```

Kiểm thêm: độ dài 64 000 mẫu, đỉnh ≤ 1,0, và `mixture ≈ speech + noise` (sai số do chia cho đỉnh phải áp cho cả ba).

## 7. Bẫy thường gặp

| Bẫy | Hậu quả |
|---|---|
| Chuẩn hoá log-mel nhiễu bằng thống kê riêng của nó | mất độ lớn tương đối giữa nhiễu và tiếng nói; `L_front` mất nghĩa. Dùng mean/std của mixture |
| Nhân mask lên log-mel đã chuẩn hoá | không có ý nghĩa vật lý; mask phải nhân trên mel công suất tuyến tính, trước log |
| Sinh mix online | `p_t` cache theo `mix_id` không còn khớp |
| Đo SNR trên đoạn có khoảng lặng | `P_s` thấp giả; dùng cùng định nghĩa RMS toàn đoạn như bộ sinh |
| Chia split theo mẫu thay vì theo clip nhiễu | rò thông tin qua các mix cùng nhiễu |
| Không căn RMS nhiễu ước lượng trước BEATs | embedding đổi theo âm lượng, tức theo SNR |
