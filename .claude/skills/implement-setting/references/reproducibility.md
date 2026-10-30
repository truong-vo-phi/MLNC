# Mã sạch và tái lập được

## Mục lục
1. Cấu hình tách khỏi mã
2. Seed và tính tất định
3. Môi trường
4. Git và nhật ký
5. Artifact, resume
6. Danh sách kiểm trước khi chạy dài

## 1. Cấu hình tách khỏi mã

Repo dùng **YAML + `BaseConfig`**, không dùng Hydra. Lý do thực tế: dataclass `frozen=True` ngăn sửa giữa chừng, khoá yaml trùng tên field nên một nơi sai là báo ngay.

```yaml
# config/kdmoe_s.yaml
model:
  d: 128
  n_vit: 2
loss:
  kd_temperature: 2.0
train:
  lr: 5.0e-4
  seed: 42
```

```python
@dataclass(frozen=True)
class KDMoEConfig(BaseConfig):
    section: ClassVar[str] = "model"
    d: int = 128
```

Quy tắc:

- Hyperparameter, URL, đường dẫn dataset: chỉ trong `config/*.yaml`. Không có đường dẫn máy (ví dụ `D:\...`) trong mã.
- Mỗi lần chạy lưu bản cấu hình đã dùng cùng kết quả (cạnh `result.csv`), để đọc lại biết đã chạy với gì.
- Đổi cấu hình bằng `dataclasses.replace(config, ...)` thay vì sửa tại chỗ (xem `run_kdmoe.py` smoke test).
- Cần đổi biến theo thí nghiệm: tạo file yaml phụ (ví dụ `kdmoe_l.yaml`), không sửa file gốc.

## 2. Seed và tính tất định

`src/utils/seed.py::set_seed` đặt `random`, `numpy`, `torch`, `torch.cuda`. Đó là mức tối thiểu. Để tái lập chặt hơn:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# tuỳ chọn, có thể báo lỗi với một số phép:
torch.use_deterministic_algorithms(True, warn_only=True)
# DataLoader
g = torch.Generator().manual_seed(seed)
DataLoader(..., generator=g, worker_init_fn=lambda w: np.random.seed(seed + w))
```

Thực tế cần biết:

- Tất định hoàn toàn trên GPU thường chậm hơn và không phải lúc nào cũng khả thi (atomic add).
- Mục tiêu hợp lý: **cùng seed thì kết quả sát nhau trong dao động nhỏ**, và báo cáo qua nhiều seed (xem skill `experiment-setting`).
- Seed phải được ghi vào kết quả, kể cả seed chia split và seed sinh mix.

## 3. Môi trường

Repo: `pyproject.toml` (cài bằng `pip install -e ".[dev]"`) và `requirements.txt` (cận dưới phiên bản). Chưa có file khoá phiên bản, conda env hay Docker.

Khi cần tái lập tuyệt đối trên máy khác:

| Cách | Khi nào |
|---|---|
| `pip freeze > output/<run>/env.txt` | Rẻ nhất, ghi lại phiên bản đã chạy |
| `pip-compile` hoặc `uv lock` | Khoá phiên bản trong repo |
| conda `environment.yml` | Cần CUDA, ffmpeg cùng phiên bản |
| Dockerfile | Chạy trên nhiều máy, CI |

Ghi lại phiên bản `torch`, `torchaudio`, CUDA. Ghi chú trong `.log/p1.md` cho thấy nâng `torch` làm đổi hành vi giữa sandbox, nên đây là nguồn lệch kết quả thật.

## 4. Git và nhật ký

- Commit nhỏ, một ý một commit; không commit dữ liệu, checkpoint (`.gitignore` đã chặn `datasets/`, `pretrained/`, `output/`).
- Kiểm tra trước commit: `ruff check`, `ruff format`, `pytest -m "not data"` (pre-commit cấu hình sẵn).
- Mỗi thay đổi thêm, sửa, xoá ghi vào `.log/p1.md` theo `.log/log-guide.md`.
- File bỏ đi chuyển vào `.bin/` giữ nguyên đường dẫn tương đối, không xoá thẳng.
- **Không đưa token vào repo.** Token Kaggle chỉ đọc từ biến môi trường `KAGGLE_ACCESS_TOKEN` (repo từng ghi chú token nằm trong `skill/guide.MD` cần thu hồi).

## 5. Artifact, resume

Cấu trúc đầu ra (`src/utils/paths.py::RunPaths`):

```
output/
  log/<exp>-<YYYY-MM-DD-HH-MM-SS>.log
  checkpoint/<exp>-<ts>/<model>/last_best_checkpoint.pt
  result/<exp>-<ts>/result.csv  (+ history/*.json)
  figure/<exp>-<ts>/*.png
```

- Có `resume.pt` mỗi epoch (optimizer, scheduler, EMA, best, stage). Resume phải kiểm `stage` khớp.
- Sandbox có giới hạn thời gian (molab 12 giờ): thiết kế để chạy lại từ `resume.pt` trên Drive.
- Đồng bộ lên Drive bằng `scripts/drive_sync.py`, kiểm md5 sau khi tải.

## 6. Danh sách kiểm trước khi chạy dài

- [ ] Smoke test 1 epoch với vài clip chạy qua (không NaN, loss giảm).
- [ ] Overfit được một batch nhỏ (xem `debugging.md`).
- [ ] Seed, cấu hình, phiên bản dữ liệu được ghi vào kết quả.
- [ ] Checkpoint và resume đã thử (dừng giữa chừng rồi chạy lại).
- [ ] Dung lượng đĩa đủ cho checkpoint và cache.
- [ ] Không có token hoặc đường dẫn máy trong mã.
