# Triển khai / resume huấn luyện FRCRN trên molab (dùng ở thiết bị khác)

Hướng dẫn này được rút ra từ lần resume ngày 2026-10-20. Không chứa token. Mọi token lấy từ nơi của bạn, không ghi vào tệp.

## Thông tin lần chạy hiện tại
- Chế độ: `--noise-source mixdataset --target speech` (mix-dataset mixture + clean, target speech), 40 epoch, early stop patience 15.
- Thư mục Drive (nơi chứa trạng thái để nối tiếp): `FA26_Capstone/Truong/frcrn-mix-speech`
  - `frcrn/resume.pt`: trọng số, optimizer, lịch sử epoch. Script tự khôi phục và chạy tiếp.
  - `frcrn/best.pt` và `frcrn/best_e<epoch>_<UTC>.pt`: bản tốt nhất và các bản lịch sử.
  - `run_manifest.json`: tiến độ và mốc thời gian từng epoch.
  - `frcrn/result.json`: chỉ có khi chạy xong. Có tệp này thì coi như đã xong; muốn chạy lại từ đầu phải dùng thư mục Drive mới.
- Tại thời điểm lưu: epoch 20 đã có `best_e020`; epoch tốt nhất theo `noise_si_snr_hi` được chọn làm `best.pt`.
- Gói code đúng với lần chạy: `output/molab-runs/frcrn-noise-20261020T132727Z/code.tar.gz` (kèm `record.json` và `REDEPLOY.md` tự sinh).

## Điều kiện trên thiết bị mới
1. Clone repo, có `scripts/molab_deploy.py`. Nếu không có repo, chép nguyên thư mục `output/molab-runs/frcrn-noise-20261020T132727Z/` (chứa `code.tar.gz`) cùng `scripts/molab_deploy.py` và `scripts/run_manifest.py`.
2. Python có `requests`.
3. Windows: đặt `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8` trước khi chạy `scripts.molab_deploy`, nếu không in tiếng Việt sẽ lỗi `UnicodeEncodeError` (cp1252).
4. Token Kaggle của bạn (chỉ cần khi sandbox chưa có dữ liệu) và quyền đăng nhập Google Drive (tài khoản có thư mục `FA26_Capstone/Truong`).

## Các bước
### 1. Tạo sandbox molab mới và mở notebook
Mở notebook trên molab, giữ tab mở (cần đúng một phiên marimo đang mở). Lấy URL sandbox (`https://sb-xxxx.sb.molab.run/`) và token marimo của nó.

### 2. Đăng nhập Google Drive trong notebook
Chạy cell đăng nhập Drive (`GoogleDriveFileSystem(..., auth_kwargs={"use_local_webserver": False})`). Cell in ra một URL Google; mở, cấp quyền, dán mã lại. Token được cache trên sandbox cho tiến trình nền dùng.

### 3. Đẩy code lên sandbox
Sandbox mới chỉ có `scripts/`, `src/`, `notebook.py`, `pyproject.toml`; thiếu `config/` (lỗi `FileNotFoundError: /marimo/config/url.yaml`) và các module `soundfile`, `torchaudio`, `timm`. Lệnh `push` xử lý cả hai:

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export MARIMO_TOKEN=<token marimo của sandbox>      # không đưa token vào dòng lệnh
python -m scripts.molab_deploy push \
  --url <URL sandbox> \
  --archive output/molab-runs/frcrn-noise-20261020T132727Z/code.tar.gz
```
- Không dùng `--force` trừ khi muốn ghi đè tệp đã khác trên sandbox. Mặc định chỉ ghi tệp còn thiếu hoặc khớp và báo tệp khác nội dung.
- `push` cài `soundfile`, `torchaudio`, `timm` bằng `uv pip install --no-deps`, ghim `torchaudio` đúng phiên bản `torch` của sandbox (đã thử: torch 2.11.0+cu130 với torchaudio 2.11.0). Không dùng `pip install torchaudio` thường vì sẽ kéo theo bản torch khác.
- `push` cũng tải checkpoint gốc về `/marimo/pretrained/last_best_checkpoint.pt` (khoảng 161 MB).
- `pesq` và `pystoi` không có trong `push`; chỉ cần khi tính điểm cảm nhận (`perceptual_scores` trong `scripts/sweep_frcrn.py`). Cài khi cần:
  `uv pip install --no-deps --python <python của kernel> pesq pystoi`.

### 4. Khởi động / resume
Có hai cách, cùng kết quả:

**A. Từ máy local bằng CLI**
```bash
export KAGGLE_ACCESS_TOKEN=<token Kaggle>     # chỉ cần nếu sandbox chưa có dữ liệu
python -m scripts.molab_deploy launch --url <URL sandbox> --kaggle -- \
  --noise-source mixdataset --target speech --use-drive \
  --drive-dir FA26_Capstone/Truong/frcrn-mix-speech \
  --work-dir /marimo/output/frcrn-noise --epochs 40
python -m scripts.molab_deploy status --url <URL sandbox> --work-dir /marimo/output/frcrn-noise
```

**B. Từ notebook (các cell sẵn có)**
1. Nhập token Kaggle ở ô "Kaggle token" (cell `vblA`).
2. Cell cài đặt: `ft_source = "mixdataset"`, `ft_full = True`, `ft_use_drive = True`, `ft_drive_dir = "FA26_Capstone/Truong/frcrn-mix-speech"`.
3. Chạy cell định nghĩa `run_all()`, rồi cell cuối với `run_training = True`. Cell này khởi động `scripts.run_frcrn_noise` chạy nền và tách khỏi trình duyệt, nên tắt máy hoặc hết pin không làm dừng huấn luyện.

Script dùng cùng `--drive-dir` nên tự đọc `resume.pt` trên Drive và nối tiếp từ epoch đã xong. Lần đầu tải dữ liệu từ Kaggle có thể mất thời gian.

### 5. Theo dõi
- `status` ở trên, hoặc cell trạng thái trong notebook (cell có `ft_refresh`).
- Log trên sandbox: `/marimo/output/frcrn-noise/train.log` (số liệu từng epoch) và `stdout.log` (thanh tiến trình, lỗi). `stdout.log` mở chế độ append, nên cuối tệp có thể còn lỗi của lần chạy trước; so với thời gian hiện tại.
- Tiến trình sống: `launcher.pid` trong thư mục làm việc.

## Lỗi đã gặp và cách xử lý
| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: soundfile` | Sandbox mới thiếu module | `push` (bước 3) hoặc cài `--no-deps` như trên |
| `FileNotFoundError: /marimo/config/url.yaml` | Sandbox không có thư mục `config/` | `push` với gói code |
| `UnicodeEncodeError ... cp1252` khi chạy `molab_deploy` trên Windows | Console cp1252 | `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` |
| Mọi cell `stale`, mất biến `ft_*` | Kernel khởi động lại | Chạy lại cell cài đặt và cell `run_all`; huấn luyện nền không bị ảnh hưởng nếu pid còn sống |
| `run_all()` trả `already running` | pid trong `launcher.pid` còn sống | Không cần làm gì; xem trạng thái |

## Lưu ý an toàn
- Không ghi token marimo, Kaggle hay Drive vào tệp, notebook hay lịch sử git. Dùng biến môi trường.
- Sandbox molab có thời hạn. Dựa vào Drive để giữ trạng thái; `resume.pt` được sao lưu sau mỗi epoch.
- Không chạy hai tiến trình huấn luyện cùng `--drive-dir`: chúng sẽ ghi đè `resume.pt` của nhau.
