# Tiêu chí đánh giá cấu trúc repo Python

Mỗi tiêu chí chấm 0 đến 3: 0 chưa có, 1 có sơ sài, 2 đạt, 3 đạt và có kiểm tra tự động.
Tổng tối đa 33. Quy tắc cho agent nằm ở [rules.MD](../rules.MD).

| # | Nhóm | Tiêu chí đạt (mức 2) | Trước | Sau |
|---|------|----------------------|:-----:|:---:|
| 1 | Đóng gói | `pyproject.toml`, cài được bằng `pip install -e .`, mọi thư mục mã là package (`__init__.py`) | 0 | 3 |
| 2 | Phụ thuộc | Khai báo có cận phiên bản, tách runtime và dev | 1 | 2 |
| 3 | Cấu hình | Siêu tham số và đường dẫn nằm trong yaml, nạp qua `BaseConfig`, không hard-code đường dẫn máy | 2 | 3 |
| 4 | Tài liệu | README: mục đích, cấu trúc, cài đặt, cách chạy, định dạng dữ liệu | 0 | 2 |
| 5 | Kiểm thử | Test pytest chạy không cần dữ liệu, test cần dữ liệu có marker và tự bỏ qua | 1 | 3 |
| 6 | Chất lượng mã | Linter và formatter có cấu hình, chạy sạch | 0 | 3 |
| 7 | Tự động hoá | pre-commit và CI chạy lint + test | 0 | 2 |
| 8 | Vệ sinh VCS | `.gitignore` đủ, `.gitattributes`, `.editorconfig`, không commit dữ liệu/checkpoint | 1 | 2 |
| 9 | Phân tách | Thư viện (`src`), script, notebook, dữ liệu và artifact tách riêng; không file/folder thừa | 2 | 3 |
| 10 | Tái lập | Seed, log, checkpoint, kết quả theo cấu trúc `output/` có timestamp, smoke test, chạy tiếp được | 2 | 3 |
| 11 | OOP nhất quán | Mọi model, loss, trainer, config, dataset thuộc một họ lớp cơ sở, cùng giao diện, có test | 0 | 3 |
| | **Tổng** | | **9** | **29** |

## Chưa đạt mức tối đa, cần quyết định của chủ repo

- **LICENSE:** chưa chọn giấy phép. Nên chọn trước khi chia sẻ repo.
- **Commit đầu tiên:** repo chưa có commit nào. Cần commit và đẩy lên remote thì CI mới chạy.
- **Mã nạp dữ liệu nằm trong `scripts/`:** về cấu trúc nên là `src/data/`. Giữ nguyên để notebook không phải đổi import.
- **Token trong `skill/guide.MD`:** file có chứa Kaggle access token ghi thẳng. Nên thu hồi token, tạo token mới và
  chỉ đọc từ biến môi trường `KAGGLE_ACCESS_TOKEN`.
- **Kiểm tra kiểu:** chưa dùng `mypy`/`pyright`.
- **Phụ thuộc chưa khoá phiên bản:** nên thêm file lock (`pip-compile` hoặc `uv lock`).
- **Test với checkpoint thật:** chưa có test tự động cho việc nạp trọng số pretrain (cần tải 363 MB và 161 MB). Đã kiểm tra
  thủ công: `load_pretrained` của BEATs và FRCRN đều trả `LoadReport` sạch.

## Khác biệt FRCRN so với bài báo (arXiv 2206.07293)

- Bài báo: một CRED. Bản phát hành (đang dùng, khớp checkpoint): hai UNet xếp tầng, `mask = tanh(u1) + tanh(u2(u1))`.
- Bài báo: cửa sổ 20 ms, bước 10 ms, STFT length 640 ở 16 kHz. Bản phát hành: `win_len=640`, `win_inc=320`.
- Bài báo: `L = L_SI-SNR + L_Mask` (lambda = 1). Bản phát hành: loss mask nhân thêm `fft_len`.
