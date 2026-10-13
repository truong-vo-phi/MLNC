# MLNC

Phân loại nhiễu dưới giọng nói: **FRCRN** (tách nhiễu bằng masking) + **BEATs** (phân loại) làm teacher, chưng cất vào
student **KD-MoE** nhỏ gọn; huấn luyện trên AudioSet-40l và Mixture-36l.

## Cấu trúc

```
MLNC/
├── config/                 # hyperparameters và đường dẫn (yaml)
│   ├── beat.yaml           # BEATs: kiến trúc, frontend, train, loss
│   ├── kdmoe_s.yaml        # KD-MoE cấu hình S (d 128, ~1.25M tham số), kèm M/L
│   ├── group_map.json      # ánh xạ mid -> label_index -> nhóm expert E0-E3 (kèm membership.npy)
│   ├── vlsp.yaml           # tỉ lệ chia train/validation/test cho VLSP-100h
│   ├── frcrn.yaml          # FRCRN: model, loss, fine-tune
│   ├── url.yaml            # URL checkpoint pretrain và đường dẫn dataset
│   └── audioset_40_labels.csv  # 40 nhãn AudioSet, dùng khi dataset thiếu metadata/40_noise_labels.csv
├── src/
│   ├── core/               # lớp cơ sở: BaseConfig, BaseModel, LoadReport
│   ├── model/beat/         # BEATs base (patch embed, encoder, attention, head)
│   ├── model/frcrn/        # FRCRN (UNet phức x2, STFT, separator)
│   ├── model/kdmoe/        # KD-MoE student lai cascade (log-mel, khối lai, cổng, ViT, BiGRU, router, 4 expert)
│   ├── loss/               # base + cross_entropy, frcrn, kdmoe (RouterKD, Front, Consistency)
│   ├── train/              # Trainer (template method) + BEATTrainer, FRCRNTrainer, KDMoETrainer (2 giai đoạn), config
│   ├── utils/              # logger, seed, checkpoint, paths, plots, EMA
│   └── metrics.py          # ClassificationMetrics
├── scripts/                # nạp dataset, checkpoint pretrain, group map, teacher cache (FRCRN -> BEATs -> probe), upload best.pt lên Google Drive (hàm, gọi từ notebook)
├── notebook/               # kd-moe.ipynb: pipeline đầy đủ (tải dữ liệu, nạp pretrain, fine-tune FRCRN -> BEATs, teacher p_t, student KD-MoE 2 stage, 9 metrics)
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
| `vlsp-100h` | chỉ cần file audio (wav/flac) ở bất kỳ thư mục con nào; split chia theo hash đường dẫn, tỉ lệ ở `config/vlsp.yaml` |
| `beats-pretrain`, `frcrn-pretrain` | URL checkpoint pretrain (HuggingFace) |
| `kaggle-*` | slug dataset Kaggle, dùng khi dataset chưa có (token đọc từ biến môi trường `KAGGLE_ACCESS_TOKEN`) |

Biến môi trường `MLNC_AUDIOSET_40L`, `MLNC_MIXTURE_36L` và `MLNC_VLSP_100H` ghi đè đường dẫn trong `url.yaml`. Cell nạp dataset gọi `ensure_dataset`:
nếu đường dẫn trong `url.yaml` không tồn tại thì dùng hoặc tải dataset vào `datasets/` rồi tự đặt biến môi trường.

## Chạy

Mở [notebook/kd-moe.ipynb](notebook/kd-moe.ipynb) (hoặc `kd-moe.ipynb.py` với marimo):

1. Tải Mixture-36l / AudioSet-40l / VLSP-100h từ Kaggle (tự bỏ qua nếu đã có).
2. Nạp pretrain FRCRN (Google Drive) + BEATs (HuggingFace), đóng băng.
3. Fine-tune FRCRN (SI-SNR) rồi BEATs (mask bằng FRCRN, chọn theo `macro_f1`).
4. Teacher p_t: embedding BEATs + linear probe cross-fit 5 fold, lưu `teacher_{split}.npz`.
5. Student KD-MoE: stage 1 (`L_front`) rồi stage 2 (CE + KD router + front + consistency).

Mọi giai đoạn có resume theo epoch và sao lưu `best.pt`/`result.json` lên Google Drive.

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
