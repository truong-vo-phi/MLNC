# rules.MD

Quy tắc bắt buộc cho agent khi viết code, tạo file hoặc folder trong repo MLNC.
Mục tiêu cao nhất: **nhất quán**. Khi có hai cách làm, chọn cách repo đang dùng.

## 0. Trước khi viết
1. Đọc file liên quan và một file tương tự đã có trước khi tạo file mới. Không đoán tên API.
2. Tái dùng lớp cơ sở, hàm tiện ích và config có sẵn. Chỉ viết mới khi không có.
3. Không tạo file hoặc folder rỗng. Không tạo folder cấp cao nhất mới nếu chưa cập nhật mục 1 và `README.md`.
4. Làm đúng yêu cầu, không thêm tính năng ngoài yêu cầu.
5. Nếu không được yêu cầu, chỉ được tuỳ chỉnh trên hyperparameters.

## 1. Vị trí file

| Loại | Nơi đặt |
|---|---|
| Model | `src/model/<ten_model>/` (mỗi file một vai trò: `config`, `backbone`, `head`, ...) |
| Loss | `src/loss/<ten>.py`, đăng ký trong `src/loss/__init__.py` |
| Trainer và config huấn luyện | `src/train/` |
| Lớp cơ sở dùng chung | `src/core/` |
| Tiện ích (logger, seed, checkpoint, paths, plots) | `src/utils/` |
| Metrics | `src/metrics.py` |
| Hàm nạp dataset, tải pretrain, tải dữ liệu | `scripts/` (chỉ hàm, không `main()`, không `ArgumentParser`) |
| Hyperparameters, URL, đường dẫn dataset | `config/*.yaml` |
| Notebook | `notebook/` |
| Test | `tests/test_<ten>.py` |
| Tài liệu | `doc/` |
| Trọng số pretrain tải về | `pretrained/` (không đưa lên git) |
| Dataset tải về từ Kaggle | `datasets/` (không đưa lên git) |
| Đầu ra khi chạy | `output/` (xem mục 5) |

- Mỗi thư mục chứa mã Python phải có `__init__.py`.
- Không để dữ liệu, trọng số, log trong `src/`, `scripts/`, `config/`.

## 2. Đặt tên
- File, hàm, biến: `snake_case`. Lớp: `PascalCase`. Hằng số: `UPPER_SNAKE`.
- Khoá yaml trùng tên field của dataclass tương ứng.
- Trong notebook (marimo/Jupyter dùng chung): mỗi tên public chỉ do một cell định nghĩa; biến lặp và biến tạm đặt tiền tố `_`; logic lặp đặt trong hàm.
- Tên thí nghiệm dùng chữ thường, gạch ngang: `frcrn-beat`.
- Tên kịch bản: `<nhom_snr>_<mask|nomask>` (ví dụ `low_snr_mask`).
- Thuộc tính cấu hình của mọi model là `self.config` (không dùng `cfg`).

## 3. Thiết kế hướng đối tượng (bốn tính chất)

**Kế thừa: mọi thành phần mới phải thuộc một họ lớp đã có.**

| Thành phần | Lớp cơ sở | Phải có |
|---|---|---|
| Cấu hình | `src.core.BaseConfig` (`@dataclass(frozen=True)`) | `section` (mục trong yaml), giá trị mặc định |
| Model | `src.core.BaseModel` | `from_yaml`, `convert_checkpoint`, `self.config` |
| Loss học có nhãn | `SupervisedLoss` | `name`, `forward(prediction, target)` |
| Loss chưng cất | `DistillationLoss` | `name`, `forward(student, teacher)` |
| Loss tách nguồn | `SeparationLoss` | `name`, `forward(mixture, target, estimate, mask)` |
| Huấn luyện | `src.train.Trainer` | `selection_metric`, `configure_optimizer`, `training_step`, `evaluate` |
| Dataset | `AudioDataset` hoặc `LabeledAudioDataset` | `dataset_key`, `__len__`, `__getitem__` |

**Đa hình: cùng một giao diện cho mọi lớp con.**
- `model.load_pretrained(path)` luôn trả về `LoadReport`. Không viết hàm nạp trọng số riêng.
- `Model.from_yaml(path)` luôn dựng model từ yaml.
- `model(...)` trả về `NamedTuple` có tên trường (`BEATOutput`, `FRCRNOutput`), không trả dict hoặc tuple trần.
- `trainer.fit(train_loader, val_loader)` trả về `TrainingResult`; `trainer.evaluate(loader)` trả về `dict[str, float]`.
- Item của dataset luôn là `AudioItem` với khoá cố định: `waveform` (đầu vào model), `target`, `snr_db`, `sample_id`, `noise`, `clean`.

**Trừu tượng: logic dùng chung nằm ở lớp cơ sở.**
- Vòng lặp epoch, early stopping, lưu checkpoint tốt nhất nằm trong `Trainer.fit`. Lớp con chỉ ghi đè hook (`begin_epoch`, `observe`, `epoch_summary`, `baseline`, `on_no_improvement`).
- Không sao chép vòng huấn luyện sang file khác.

**Đóng gói: giấu chi tiết nội bộ.**
- Thành viên nội bộ bắt đầu bằng `_`. Chỉ lộ API cần thiết qua `__init__.py` và `__all__`.
- Config là `frozen`; đổi giá trị bằng `dataclasses.replace`, không gán trực tiếp.
- Tham số có thể đổi (lr, epoch, batch size, SNR) nằm trong config, không rải trong hàm.

## 4. Cấu hình và đường dẫn
- Không hard-code số siêu tham số, URL, đường dẫn tuyệt đối trong `.py`. Dùng `config/*.yaml`.
- Đường dẫn dataset đọc từ `config/url.yaml` qua `resolve_dataset_root`; môi trường khác (molab, Colab) ghi đè bằng biến `MLNC_<KEY>` (xem `dataset_env_name`), không sửa `url.yaml` theo máy.
- Hyperparameters huấn luyện nằm ở mục `train:` của yaml tương ứng và nạp bằng `<Config>.from_yaml`.
- Giá trị chưa xác nhận trong yaml đánh dấu `# TODO`.

## 5. Cấu trúc đầu ra (theo `skill/guide.MD`)
Mọi lần chạy dùng `src.utils.paths.RunPaths`, không tự ghép đường dẫn.

```
output/log/<experiment>-<YYYY-MM-DD-HH-MM-SS>.log
output/checkpoint/<experiment>-<YYYY-MM-DD-HH-MM-SS>/<model>/last_best_checkpoint.pt
output/result/<experiment>-<YYYY-MM-DD-HH-MM-SS>/result.csv
output/result/<experiment>-<YYYY-MM-DD-HH-MM-SS>/history/<ten>.json
output/figure/<experiment>-<YYYY-MM-DD-HH-MM-SS>/*.png
```

- Dấu thời gian theo ví dụ trong guide: `experiment-01-2026-10-02-22-36-09` (năm-tháng-ngày-giờ-phút-giây).
- Smoke test ghi vào `output/test/` với cùng cấu trúc.
- `result.csv` là bảng kết quả cuối (mỗi dòng một kịch bản, cột là metrics). Lịch sử theo epoch lưu ở `history/`.
- Chạy tiếp một lần chạy cũ bằng `RESUME_RUN_ID`, không tạo thư mục mới.

## 6. Notebook
- Thứ tự cell cố định: (1) `PIPELINE_NAME`, `RUN_TRAINING`, đường dẫn đầu ra, logger; (2) tham số cấu hình; (3) nạp dataset; (4) nạp pretrain; (5) dựng kiến trúc; (6) huấn luyện và đánh giá; thêm cell hiển thị cấu hình và biểu đồ ở cuối.
- `RUN_TRAINING = True` chạy thật, `False` chạy smoke test nhỏ.
- Notebook chỉ gọi hàm và lớp trong `src/` và `scripts/`. Không viết lại logic huấn luyện trong cell.
- Cell nạp dataset gọi `scripts.download_dataset.ensure_dataset`; token đọc từ biến môi trường `KAGGLE_ACCESS_TOKEN`.
- Tiến trình dùng `logger` cho mốc chính, `print` và `tqdm` cho tiến độ chi tiết.
- Không commit output của cell (dùng `nbstripout`).

## 7. Phong cách code
- Python >= 3.10, `from __future__ import annotations`, có type hint cho hàm công khai.
- Không viết comment hoặc docstring trong file `.py` (ngoại lệ: chú thích shape ngắn trong kiến trúc đã có sẵn).
- Hàm ngắn, một trách nhiệm. Không `print` trong `src/` (dùng logger).
- Lỗi đầu vào sai: ném `ValueError`, `KeyError`, `FileNotFoundError` rõ ràng; không nuốt lỗi.
- Format và lint bằng ruff: `ruff check . --fix && ruff format .` (dòng tối đa 120).

## 8. Kiểm thử
- Mỗi lớp hoặc hàm công khai mới có test trong `tests/`.
- Test không phụ thuộc dữ liệu cục bộ hoặc mạng. Test cần dữ liệu gắn `@pytest.mark.data` và tự bỏ qua khi thiếu.
- Sau thay đổi mã ML: chạy một smoke pass nhỏ (vài mẫu, 1 epoch), kiểm tra shape, loss hữu hạn, khoá metrics.
- Trước khi báo xong phải chạy được: `ruff check .`, `ruff format --check .`, `pytest -m "not data"`.

## 9. Bảo mật
- Không ghi token, mật khẩu, khoá API vào file trong repo (kể cả notebook, md, yaml). Đọc từ biến môi trường.
- Không đọc hoặc in nội dung `.env`.
- `torch.load` chỉ với file đáng tin. Đặt `weights_only = True` ở lớp con của `BaseModel` khi checkpoint là state dict thuần (như FRCRN); mặc định `False` vì checkpoint BEATs chứa cấu hình.

## 10. Git
- Không commit: `pretrained/`, `datasets/`, `output/test/`, `output/log/`, `output/checkpoint/`, file `.pt`, dữ liệu.
- Không commit hoặc push khi người dùng chưa yêu cầu.
- Thêm phụ thuộc mới vào `requirements.txt` (runtime) hoặc `requirements-dev.txt` (dev).

## 11. Danh sách kiểm tra trước khi báo xong
- [ ] Lớp mới kế thừa đúng lớp cơ sở ở mục 3.
- [ ] Không hard-code siêu tham số, đường dẫn, secret.
- [ ] File nằm đúng thư mục ở mục 1, có `__init__.py` nếu là package.
- [ ] Đầu ra qua `RunPaths`.
- [ ] Có test; `ruff` và `pytest -m "not data"` đều qua.
- [ ] README hoặc `rules.MD` được cập nhật nếu thêm folder hoặc quy ước mới.
