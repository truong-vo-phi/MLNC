# MLNC

Phân loại nhiễu dưới giọng nói: **FRCRN** (tách nhiễu bằng masking) + **BEATs** (phân loại), huấn luyện trên
AudioSet-40l và Mixture-36l.

## Cấu trúc

```
MLNC/
├── config/                 # hyperparameters và đường dẫn (yaml)
│   ├── beat.yaml           # BEATs: kiến trúc, frontend, train, loss
│   ├── frcrn.yaml          # FRCRN: model, loss, fine-tune
│   ├── url.yaml            # URL checkpoint pretrain và đường dẫn dataset
│   └── audioset_40_labels.csv  # 40 nhãn AudioSet, dùng khi dataset thiếu metadata/40_noise_labels.csv
├── src/
│   ├── core/               # lớp cơ sở: BaseConfig, BaseModel, LoadReport
│   ├── model/beat/         # BEATs base (patch embed, encoder, attention, head)
│   ├── model/frcrn/        # FRCRN (UNet phức x2, STFT, separator)
│   ├── loss/               # base + cross_entropy, kd, crd, frcrn
│   ├── train/              # Trainer (template method) + BEATTrainer, FRCRNTrainer, config
│   ├── utils/              # logger, seed, checkpoint, paths, plots
│   └── metrics.py          # ClassificationMetrics
├── scripts/                # nạp dataset và checkpoint pretrain (hàm, gọi từ notebook)
├── notebook/               # beat-v1.ipynb (FRCRN fine-tune + BEATs), beat-v2.ipynb (BEATs + masking từ pretrained/frcrn_best.pt)
├── tests/                  # pytest
├── doc/                    # tài liệu, tiêu chí đánh giá repo
├── skill/guide.MD          # quy định cấu trúc đầu ra và notebook
├── rules.MD                # quy tắc cho agent khi viết code, tạo file/folder
├── pretrained/             # trọng số pretrain tải về (không đưa lên git)
├── datasets/               # dataset tải về từ Kaggle khi chưa có sẵn (không đưa lên git)
└── output/
    ├── log/                # <experiment>-<YYYY-MM-DD-HH-MM-SS>.log
    ├── checkpoint/         # <experiment>-<timestamp>/<model>/last_best_checkpoint.pt
    ├── result/             # <experiment>-<timestamp>/result.csv (+ history/*.json)
    ├── figure/             # <experiment>-<timestamp>/*.png
    └── test/               # lịch sử smoke test, cùng cấu trúc như trên
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Nếu cần GPU, cài PyTorch bản CUDA trước theo hướng dẫn của PyTorch rồi mới chạy lệnh trên.

## Dữ liệu

Đường dẫn dataset khai báo trong [config/url.yaml](config/url.yaml):

| Khoá | Nội dung |
|------|----------|
| `audioset-40l` | `audio/` (wav) và `metadata/` (manifest, `40_noise_labels.csv`) |
| `mixture-36l` | `train/`, `validation/`, `test/` (mỗi split có `manifest.csv`), `labels.txt` |
| `beat-pretrain`, `frcrn-pretrain` | URL checkpoint pretrain |
| `kaggle-*` | slug dataset Kaggle, dùng khi dataset chưa có (token đọc từ biến môi trường `KAGGLE_ACCESS_TOKEN`) |

Biến môi trường `MLNC_AUDIOSET_40L` và `MLNC_MIXTURE_36L` ghi đè đường dẫn trong `url.yaml`. Cell nạp dataset gọi `ensure_dataset`:
nếu đường dẫn trong `url.yaml` không tồn tại thì dùng hoặc tải dataset vào `datasets/` rồi tự đặt biến môi trường.

## Chạy

Mở [notebook/beat-v1.ipynb](notebook/beat-v1.ipynb):

- `RUN_TRAINING = False`: smoke test nhanh, lịch sử lưu vào `output/test/`.
- `RUN_TRAINING = True`: chạy thật, 4 nhóm SNR (low, mid, high, all) x (có/không masking) = 8 kịch bản, lưu vào `output/`.
- `RESUME_RUN_ID = "<experiment>-<timestamp>"`: chạy tiếp một lần chạy trước (bỏ qua phần đã xong).

## Kiểm thử và chất lượng mã

```bash
pytest -m "not data"     # không cần dataset
pytest                   # gồm cả test cần dataset cục bộ
ruff check . && ruff format --check .
pre-commit install
```

## Lưu ý

- UNet của FRCRN cố định `fft_len=640` để khớp checkpoint pretrain.
- Bản FRCRN ở đây theo bản phát hành (hai UNet xếp tầng), khác bài báo gốc một số chi tiết, xem
  [doc/project-standards.md](doc/project-standards.md).
