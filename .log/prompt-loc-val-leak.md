# Prompt: lọc val/test khỏi các ytid mà model pretrained AudioSet đã thấy

Dán nguyên khối dưới đây cho một phiên Claude Code mới (thư mục làm việc `D:\PROJECT\MLNC`).

---

## Bối cảnh

Repo MLNC phân loại nhiễu dưới giọng nói (36 lớp). Dữ liệu `mix-dataset` nằm ở `D:\content\mix-dataset` (các thư mục `train`, `validation`, `test`, mỗi thư mục có `manifest.csv` với cột `sample_id`, `noise_ytid`, `label_names`, `target_snr_db`, ...). Nhiễu lấy từ AudioSet, danh sách ytid chính thức ở `D:\content\AudioSet\metadata`: `eval_segments.csv`, `balanced_train_segments.csv`, `unbalanced_train_segments.csv` (bỏ các dòng bắt đầu bằng `#`, ytid là cột đầu).

Đã kiểm tra: không có ytid hay `speech_id` trùng giữa train, val, test. Nhưng khoảng 51% ytid của val (298/579) và 49% của test (279/574) thuộc `balanced_train_segments`, tức nằm trong dữ liệu train AudioSet mà các model pretrained (BEATs AS2M, CED, PANNs CNN10, EfficientAT) có thể đã thấy. Chỉ phần thuộc `eval_segments` là sạch.

## Việc cần làm

1. **Tạo danh sách lọc.** Viết `scripts/filter_pretrain_leak.py` đọc `validation/manifest.csv` và `test/manifest.csv`, giữ sample có `noise_ytid` thuộc `eval_segments` và không thuộc `balanced_train` hay `unbalanced_train`. Ghi `validation_clean.csv` và `test_clean.csv` (cột `sample_id`, `noise_ytid`) vào `D:\content\mix-dataset\`. Không sửa manifest gốc.
   - Kết quả mong đợi: val còn **3064/6480** sample (281 ytid), test còn **3253/6480** (295 ytid). Nếu số khác thì dừng và báo.
2. **In thống kê:** số sample theo từng lớp và từng mức SNR sau lọc. Cả 36 lớp phải còn mẫu; báo các lớp dưới 50 mẫu (val: Shatter 37, Clarinet 46).
3. **Cho loader dùng danh sách này.** Thêm tham số tuỳ chọn (ví dụ `sample_ids_file`) vào `Mixture36Dataset` trong `scripts/load_dataset.py` (và bản trong `imp/kd-v4/scripts/` nếu hai bản đang đồng bộ), mặc định `None` để không đổi hành vi cũ. Thêm test nhỏ bằng dữ liệu tổng hợp.
4. **Đánh giá lại.** Với các checkpoint đã có (xem `output/` và `pretrained/`), tính lại metric trên val/test đã lọc và so với toàn bộ: accuracy, balanced accuracy, macro-F1, mAP, theo từng SNR. Ghi bảng "đầy đủ vs đã lọc vs phần bị loại".
5. **Kết luận.** Nếu điểm trên phần bị loại (balanced) cao hơn rõ phần sạch, đó là dấu hiệu leak từ pretrain. Báo mức chênh, đừng suy diễn khi chưa đủ mẫu.

## Ràng buộc

- Không huấn luyện lại. Việc suy luận nặng chỉ chạy local khi tôi đồng ý; nếu cần GPU, hỏi trước.
- Chọn epoch/siêu tham số bằng val gốc như cũ; chỉ dùng tập đã lọc để báo cáo.
- Ghi checkpoint nào đã dùng (đường dẫn và md5) và checkpoint pretrained nào (bản SSL `BEATs_iter3_plus_AS2M.pt` hay bản fine-tune AS2M) vào kết quả.
- Ghi lại thay đổi vào `.log/p1.md` theo định dạng hiện có.
- Chạy thử trên 64 mẫu trước khi chạy toàn bộ.

## Kết quả cần giao

Hai file `*_clean.csv`, script lọc, thay đổi loader kèm test, bảng so sánh điểm, và 3-5 dòng kết luận có hay không có leak đáng kể.
