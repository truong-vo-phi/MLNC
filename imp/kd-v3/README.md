# kd-v3: KD teacher-student (BEATs → FRCRN + MobileViT-S), snapshot từ sandbox molab

Snapshot mã và cấu hình của notebook `notebook.py` (marimo) ngày 2026-10-25, để triển khai lại khi sandbox mất. **Không gồm dữ liệu, trọng số pretrained, cache n̂ và checkpoint.**

## Nội dung

| Đường dẫn | Là gì |
| --- | --- |
| `notebook.py` | Notebook marimo (28 cell, gồm cả cell mô tả). Cell 9 (huấn luyện) đang **bật**: Run all sẽ chạy cả hàng đợi `RUNS` |
| `src/` | Mô hình, loss, trainer, metric; `src/calibration.py` là hiệu chỉnh bias theo lớp |
| `scripts/` | Tải dataset (`download_dataset.py`), nạp dữ liệu (`load_dataset.py`), đồng bộ Drive, `molab_deploy.py` |
| `config/` | YAML cấu hình (`beat.yaml`, `frcrn.yaml`, `url.yaml`...), `group_map.json`, `membership.npy` |
| `pyproject.toml`, `lock.txt` | Phụ thuộc của sandbox (marimo 0.25.2) |
| `molab_helpers/` | Script phụ khi triển khai lần đầu (cache, download, master, train, e2e) |
| `results/reference_run/` | Log và `history_*.csv` của thí nghiệm tham chiếu (cả ba loss KD, 33 epoch) |
| `diagnostics/` | Xác suất val của student epoch 27 và 33 + teacher trên n̂, ma trận nhầm lẫn val (epoch 27) |

## Cần có ngoài gói này

| Thứ | Ở đâu | Ghi chú |
| --- | --- | --- |
| Dataset `mixture-36l` (~23 GB zip, ~31 GB giải nén) | Kaggle, qua `scripts/download_dataset.py` | Cần biến môi trường `KAGGLE_ACCESS_TOKEN`. Cell 5b tải và kiểm tra manifest. **Cell 5b không tải tiếp được**: nếu bị ngắt, xoá `datasets/mix-dataset.zip.part` rồi chạy lại, hoặc tải tiếp bằng HTTP Range |
| Teacher `beats_teacher_fold0_weights.pt` (182 MB) | `pretrained/beat-ft/` ở repo local | SHA-256 bắt đầu `60a037f5111c178e` |
| FRCRN `best_e037_20261020T203958Z.pt` (58 MB) | `pretrained/FRCRN/` ở repo local | SHA-256 bắt đầu `d398860fe7aa041b` |
| Cache n̂ của FRCRN (zip 8,6 GB) | Drive: `FA26_Capstone/Truong/beat-mv-kd-v3/nhat/nhat_cache.zip` | Cell 6b tự khôi phục từ Drive qua `fs`; nếu thiếu sẽ tính lại bằng FRCRN (~17 phút). `fs.get_file` của gdrive_fsspec lỗi với tệp nhị phân, khôi phục cần đọc nhị phân bằng `fs.open(..., "rb")` |
| Đăng nhập Google Drive | cell 2b (liên kết đăng nhập lần đầu) | Không có thông tin đăng nhập nào trong gói |

## Triển khai lại

1. Tạo sandbox molab, đẩy mã bằng `python -m scripts.molab_deploy push --url URL --archive <code.tar.gz>` (hoặc chép `notebook.py`, `src/`, `scripts/`, `config/`, `pyproject.toml` vào `/marimo`). Token marimo đọc từ biến môi trường `MARIMO_TOKEN`, không ghi vào tệp.
2. Đặt checkpoint pretrained vào `pretrained/beat-ft/` và `pretrained/FRCRN/` (đúng đường dẫn ở cell 1).
3. Chạy cell 1-4 (cấu hình, Drive, teacher, student), rồi 5b (dataset) và 6b (cache n̂).
4. Cell 9: **kiểm tra `RUNS` ở cell 1 trước khi Run all**.

## Hàng đợi thí nghiệm (cell 1, `RUNS`)

Mặc định `no-crd`, `no-qkv`, `ce`, `hn`: 4 × 15 = 60 epoch, khoảng 3 giờ (~3 phút mỗi epoch có teacher, gồm upload Drive). Nhánh `eskd` là nhánh duy nhất không huấn luyện từ đầu (khởi tạo từ `best_state` của tham chiếu) nên được chú thích sẵn trong `RUNS`. Mỗi nhánh có `resume.pt` riêng và tên `<tên cơ sở>-<tag>-r4`, nên chạy lại sẽ nối tiếp. Nhánh ablation đặt `eval_test=False`: **chưa động tới test**; chỉ chạy test một lần cho cấu hình được chọn. Thí nghiệm tham chiếu (cfg rỗng) không nằm trong hàng đợi; bảng so sánh lấy đường cong của nó từ `resume.pt` nên cần có `output/resume/<tên tham chiếu>/resume.pt` (hoặc tải từ Drive).

Cell hiệu chỉnh bias (`run_calibration`) chưa được chạy trên student thật. Thử trên epoch 27 cho thấy macro recall chỉ tăng +0,002, nên không phải đòn bẩy chính.

## Kết quả tham chiếu (validation, thí nghiệm cả ba loss KD)

- Tốt nhất ở epoch 27: macro_acc 0,3946; epoch 33: 0,3938; train acc 0,75.
- Teacher fold 0 trên n̂ của FRCRN (validation): 0,6633; trên nhiễu sạch: 0,7402.
- Lỗi tập trung ở các nhóm lớp gần nhau (Saxophone/Clarinet/Trumpet/Flute, Glockenspiel/Chime/Electric piano, các âm xung → Gunshot, Scissors/Tearing/Writing, Microwave/Gong/Bus).
- Chi tiết các cell mô tả nằm trong `notebook.py` ("Hard-negative theo nhóm lớp dễ nhầm", "Huấn luyện và đánh giá", "Hiệu chỉnh bias theo lớp").

## Lưu ý

- Checkpoint tham chiếu cũ (định dạng chưa có đầu phân loại `att/cla`, `fmt=1`) đã được đổi tên trên Drive thành `…-qkv-crd-logit-fmt1-backup`; không dùng để nối tiếp (cell từ chối vì `fmt` khác).
- `src/calibration.py` ở local gốc (`D:\PROJECT\MLNC\src\`) có thể khác bản này; bản này là bản đã chạy trong sandbox.
- Nếu ID cell trong marimo đổi sau khi nạp lại notebook (ID theo vị trí), luôn tìm cell theo nội dung, không theo ID cũ. Một lần chạy lệnh theo ID cũ đã làm mất ba cell code (kiểm tra teacher, hàm huấn luyện, hiệu chỉnh) và phải khôi phục từ bản snapshot này.
- `RUN_TAG` (cell 1) hiện là `r6`: đổi tag để bắt đầu lại các nhánh từ epoch 1 mà không xoá gì; nhớ tắt cell 9 trước khi sửa cell 1, vì sửa cell 1 sẽ chạy lại cell 9.
