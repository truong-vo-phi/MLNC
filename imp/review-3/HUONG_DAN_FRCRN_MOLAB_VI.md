# Thí nghiệm 9 pretrained classifier trên cache FRCRN

## Dữ liệu và mục đích

Dùng `cache-nhat.zip` của `phitruong0606/mix-dataset-v2`, revision
`ea3ce2572be476fa90b5f6424ab461330bd23b39`, theo xác nhận của người dùng ngày
04/11/2026. Tên `frcrn-cache.zip` trong đoạn tải ban đầu hiện trả về 404.

Cache chứa `train.npy` (30.240 × 64.000) và `validation.npy` (3.064 × 64.000),
float32, tương ứng audio mono 16 kHz dài 4 giây. Đây là nhiễu ước lượng từ
FRCRN, không phải nhiễu sạch ground truth. Không chạy lại FRCRN để tạo cache;
không truyền gradient qua FRCRN. Chỉ fine-tune classifier.

Giữ pretrained, head 36 lớp, hyperparameter, seed, augmentation, loss và cách
chọn checkpoint giống baseline mixture. Dùng trực tiếp biên độ trong cache,
không thêm RMS normalization riêng cho lượt mask. Đầu vào thay đổi từ mixture
sang nhiễu ước lượng; không warm-start từ classifier đã fine-tune mixture.

## Kiểm tra dữ liệu

`main.py prepare` kiểm tra SHA256 ZIP, hai NPY, hai manifest và label map;
kiểm tra shape, dtype và dữ liệu hữu hạn. Chỉ lấy manifest/labels từ ZIP gốc,
không giải nén WAV của tập test. Các hàng cache ghép với manifest cùng revision.

Cache không có sample-ID sidecar. Cache đúng checksum đã được kiểm tra trước
đây ở 16 vị trí: cosine với residual FRCRN thấp nhất 0,999823. Đây là kiểm tra
mẫu, chưa phải chứng minh thứ tự toàn bộ hàng. Giới hạn này được ghi trong
`frcrn_provenance.json` và `effective_config.json` của mỗi model.

Loader từ chối manifest đổi thứ tự, cache bị thay đổi sau khi xác minh, sai
shape/dtype hoặc cache chưa có provenance. Không tự lấy file NPY đầu tiên,
không cắt bớt hàng cho khớp và không tự sinh nhãn khi thiếu manifest.

## Chạy trong Molab

Tại thư mục `MLNC/imp/review-3` trong phiên Molab có GPU:

```bash
python main.py prepare --input-kind frcrn --data-dir data --frcrn-cache data/frcrn --remove-archive
python main.py preflight --model all --config config/pretrained_frcrn.yaml --device cpu
python main.py train --model all --config config/pretrained_frcrn.yaml --device cuda --smoke
python main.py train --model all --config config/pretrained_frcrn.yaml --device cuda --drive-dir MLNC/review3_pretrained_frcrn_v2
```

Chạy đủ smoke test trước lịch đầy đủ. Đăng nhập Google Drive trong notebook
trước khi chạy lệnh cuối. Nếu phiên đổi, chuẩn bị lại dữ liệu/trọng số và đăng
nhập, dùng cùng thư mục code, config, lựa chọn `all` và Drive folder để resume.

## Đầu ra

- `logs/pretrained_frcrn/`: tiến trình, preflight, queue và bảng tổng hợp.
- `checkpoints/pretrained_frcrn/<model>/`: best.pt, last.pt, history, results,
  predictions, bảng metric theo SNR và cấu hình thực tế.
- Drive: `MLNC/review3_pretrained_frcrn_v2/<model>/`.

Đánh giá validation bằng acc, macro precision, macro recall, macro-F1,
micro-F1 và mAP tại -5/0/5/10/15/20 dB và all. Các mốc SNR là nhãn SNR của
mixture trước mask; không gọi chúng là SNR đã được đo lại sau FRCRN.

Kết quả/log/checkpoint được lưu riêng so với baseline mixture. Các metric
cuối được upload trước checkpoint resume lớn, để lỗi tải checkpoint không
ngăn việc lưu báo cáo. Backup best khi cải thiện; cặp resume epoch 1, mỗi 5
epoch và cuối lượt chạy. Kiểm tra `backup_status.json` để xác nhận.
