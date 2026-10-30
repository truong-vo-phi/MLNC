import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    _cwd = Path.cwd()
    ROOT = _cwd.parent if _cwd.name == "notebook" else _cwd
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ROOT, mo


@app.cell
def _(mo):
    mo.md(r"""
    # frcrn-verify: cách fine-tune FRCRN trên AudioSet và kiến trúc

    Notebook **mô tả và kiểm chứng**, không huấn luyện. Mặc định chỉ đọc cấu hình và mã trong repo;
    phần kiểm tra số học (đếm tham số, chạy thử một lượt xuôi trên CPU) nằm sau cờ `RUN_CHECKS = False`.

    Nhãn độ chắc chắn dùng xuyên suốt: **[đọc]** lấy từ mã hoặc tệp trong repo, **[suy luận]** suy ra từ mã,
    **[chưa kiểm chứng]** chưa có bằng chứng.

    | Mục | Nội dung |
    | --- | --- |
    | 1 | Bức tranh tổng thể của pipeline fine-tune |
    | 2 | Dữ liệu: trộn động tiếng nói (mix-dataset) với nhiễu AudioSet-40l |
    | 3 | Kiến trúc FRCRN (STFT, hai U-Net xếp tầng, mask phức, iSTFT) |
    | 4 | Hàm mất mát: mask loss + SI-SNR |
    | 5 | Vòng huấn luyện và cấu hình (đọc trực tiếp từ `config/frcrn.yaml`) |
    | 6 | Kiểm tra số học (tuỳ chọn) |
    | 7 | Những gì đã và chưa kiểm chứng |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Bức tranh tổng thể

    Giai đoạn 1 của `beat-v1`: fine-tune FRCRN (khởi tạo từ checkpoint pretrain) rồi **đóng băng**,
    dùng làm bộ lọc nhiễu cho giai đoạn 2 (BEATs). **[đọc]** `scripts/finetune.py`, notebook `beat-v1` ở commit `2ccdb45`.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart LR
          subgraph D["Dữ liệu (trộn động mỗi lần lấy mẫu)"]
            S["Tiếng nói sạch<br/>clean_path của mix-dataset"]
            N["Nhiễu AudioSet-40l<br/>đã loại ytid của validation/test"]
            M["mix_at_snr<br/>SNR ngẫu nhiên: -5, 0, 5, 10, 15, 20 dB"]
            S --> M
            N --> M
          end
          M -->|"mixture, noise, snr_db"| T["FRCRNTrainer<br/>Adam, lr 1e-4"]
          P["Pretrain<br/>FRCRN_SE_16K"] -->|"load_pretrained"| T
          T --> L["Loss = mask loss - SI-SNR"]
          L --> T
          T -->|"mỗi epoch"| V["Validation<br/>noise_si_snr"]
          V -->|"tốt hơn best"| C["best.pt<br/>+ gửi Drive"]
          V -->|"stale = 3"| H["nạp lại best<br/>lr x 0,5"]
          V -->|"stale = 15"| E["early stop"]
          C --> F["freeze()<br/>teacher cho BEATs"]
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Dữ liệu: mixture được tạo động từ AudioSet

    Lớp `DynamicMixDataset` (`scripts/load_dataset.py`). **[đọc]**

    | Thành phần | Train | Validation |
    | --- | --- | --- |
    | Tiếng nói | các `clean_path` duy nhất trong `train/manifest.csv` của mix-dataset | như vậy, trong `validation/manifest.csv` |
    | Nhiễu | `AudioSet40Dataset(split="train")` | `AudioSet40Dataset(split="validation")` |
    | Loại nhiễu | bỏ các clip có ytid nằm trong validation và test của mix-dataset (`held_out_noise`) | như train |
    | Độ dài đoạn | 2,0 s (`segment_seconds`) | 4,0 s (`eval_segment_seconds`) |
    | SNR | chọn ngẫu nhiên trong 6 mức mỗi lần lấy mẫu | như train nhưng cố định theo `seed + index` (`deterministic=True`) |
    | Nhiễu / vị trí cắt | rút ngẫu nhiên mỗi lần | cố định theo `seed + index` |
    | DataLoader | `shuffle=True`, `drop_last=True` | `shuffle=False` |

    - Số mẫu mỗi epoch = số câu sạch duy nhất của split, không phải 30 240 hàng của manifest mixture. **[đọc]** `len(...) = len(speech_paths)`.
    - Mỗi mẫu trả về `waveform` (mixture), `noise`, `snr_db`. Nhãn huấn luyện của tiếng nói được suy ra là `mixture − noise`. **[đọc]**
    - Chi tiết bên trong `mix_at_snr` (cách chuẩn hoá năng lượng khi trộn) **[chưa kiểm chứng]** trong notebook này.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart TB
          subgraph Train["Train"]
            A1["Tiếng nói: chỉ mục i"] --> R1["rng ngẫu nhiên"]
            A2["Nhiễu AudioSet train<br/>rút ngẫu nhiên"] --> R1
            R1 --> X1["mix_at_snr"]
            X1 --> O1["mixture 2 s, noise, snr_db"]
          end
          subgraph Val["Validation"]
            B1["Tiếng nói: chỉ mục i"] --> R2["rng = seed + i"]
            B2["Nhiễu AudioSet validation<br/>chọn theo rng"] --> R2
            R2 --> X2["mix_at_snr"]
            X2 --> O2["mixture 4 s, noise, snr_db<br/>giống hệt mỗi lần đánh giá"]
          end
          H["held_out_noise<br/>ytid của validation, test (mix-dataset)"] -.->|"loại bỏ"| A2
          H -.->|"loại bỏ"| B2
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Kiến trúc FRCRN

    Lớp bọc `FRCRNSeparator` chuẩn hoá biên độ rồi gọi mạng `FRCRN` (`src/model/frcrn/`). **[đọc]**

    **Luồng xuôi (waveform → waveform):**

    1. `audio_norm`: đưa mixture về mức −25 dB (hai bước: RMS, rồi RMS của các mẫu to), giữ lại hệ số `scale`.
    2. `ConvSTFT`: cửa sổ 640, hop 320, `fft_len` 640 cài bằng `conv1d`. Cho 321 bin, tách thành phần thực và ảo.
    3. `UNet` thứ nhất nhận tensor `[B, 1, 321, T, 2]` (kênh cuối là thực/ảo), trả về mask thô.
    4. `UNet` thứ hai nhận đầu ra của UNet thứ nhất. **Mask cuối = tanh(UNet1) + tanh(UNet2(UNet1))** (kiểu "tinh chỉnh chồng", đúng ý tưởng FRCRN: hai U-Net xếp tầng).
    5. Áp mask phức: `(Re·mRe − Im·mIm, Re·mIm + Im·mRe)`.
    6. `ConviSTFT` + overlap-add, cắt về đúng độ dài ban đầu, nhân lại `scale`.
    7. Nếu `target = speech` thì đầu ra là tiếng nói ŝ, nhiễu ước lượng là `mixture − ŝ`.

    **Một U-Net** (`unet.py`): `[đọc]`

    | Khối | Số lượng | Chi tiết |
    | --- | --- | --- |
    | Encoder | 7 | `ComplexConv2d` (kernel (5,2) ×6 rồi (2,2)), stride (2,1), padding (0,1), kênh 1 → 128 → … → 128; `ComplexBatchNorm2d` + `LeakyReLU` |
    | FSMN giữa encoder | 6 (từ lớp 2 trở đi) | `ComplexUniDeepFsmn_L1`, bậc nhớ 20 |
    | SE (squeeze-excitation phức) | 7 | giảm kênh ÷ 8, kênh thực/ảo gộp chéo |
    | Cổ chai | 1 | `ComplexUniDeepFsmn` hai tầng (L1, L2), kênh 128 |
    | Decoder | 7 | `ComplexConvTranspose2d`, nối skip từ encoder tương ứng; cuối cùng kênh 1 |
    | Đầu ra | 1 | `ComplexConv2d` 1×1 |

    Tách thực/ảo: mọi phép (conv, BN, FSMN, SE) đều có hai nhánh `*_re`, `*_im` kết hợp theo phép nhân phức. **[đọc]** `complex_nn.py`.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart TB
          W["waveform (mixture)"] --> N["audio_norm<br/>mức -25 dB, giữ scale"]
          N --> ST["ConvSTFT<br/>win 640, hop 320, 321 bin"]
          ST --> RI["tách thực / ảo<br/>[B,1,321,T,2]"]
          RI --> U1["UNet 1<br/>7 encoder + FSMN + SE + 7 decoder"]
          U1 --> T1["tanh"]
          U1 --> U2["UNet 2<br/>cùng cấu trúc, nhận đầu ra UNet 1"]
          U2 --> T2["tanh"]
          T1 --> SUM["cộng: mask phức"]
          T2 --> SUM
          RI --> APPLY["nhân phức với mask"]
          SUM --> APPLY
          APPLY --> IST["ConviSTFT + overlap-add"]
          IST --> FIT["cắt về độ dài gốc, nhân scale"]
          FIT --> OUT["ŝ (tiếng nói) hoặc nhiễu"]
          OUT --> NOISE["nhiễu ước lượng = mixture - ŝ"]
        """
    )
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart LR
          X["x"] --> E1["Enc 1<br/>1 → 128"]
          E1 --> E2["FSMN + Enc 2"]
          E2 --> E3["FSMN + Enc 3"]
          E3 --> E4["FSMN + Enc 4"]
          E4 --> E5["FSMN + Enc 5"]
          E5 --> E6["FSMN + Enc 6"]
          E6 --> E7["FSMN + Enc 7"]
          E7 --> BN["FSMN cổ chai (L1, L2)"]
          BN --> D1["Dec 1"]
          D1 --> D2["FSMN + SE + Dec 2"]
          D2 --> D3["..."]
          D3 --> D7["Dec 7"]
          D7 --> LIN["Conv 1x1 → mask"]
          E6 -. "skip: nối đầu ra encoder tương ứng, trước mỗi decoder kế tiếp" .-> D1
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Hàm mất mát

    `FRCRNLoss` (`src/loss/frcrn.py`). **[đọc]** Với `y` = mixture đã chuẩn hoá, `s` = mục tiêu, `ŝ` = ước lượng:

    $$\mathcal{L} = w_m \cdot N_{fft}\,\Big(\mathrm{MSE}(\hat M_{re}, M^\*_{re}) + \mathrm{MSE}(\hat M_{im}, M^\*_{im})\Big) \;-\; \overline{\mathrm{SI\text{-}SNR}(s,\hat s)}$$

    - Mask lý tưởng `M*` = tỉ số phức `S · conj(Y) / (|Y|² + ε)`, **cắt giá trị ngoài ±2 (`mask_bound`) về ±1**, STFT Hann 640/320 không đệm giữa.
    - `N_fft = 640` nhân vào MSE để cân với thang dB của SI-SNR; `w_m = mask_weight` (mặc định 1,0).
    - `SI-SNR(s, ŝ)` = $10\log_{10}\dfrac{\lVert \alpha s\rVert^2}{\lVert \hat s-\alpha s\rVert^2}$ với $\alpha = \langle \hat s, s\rangle / \lVert s\rVert^2$; trừ trung bình mỗi đoạn trước khi tính.
    - Trong huấn luyện, mục tiêu `s` là `mixture − noise` (khi `target=speech`) hoặc `noise` (khi `target=noise`), cả hai đã chia theo `scale`.
    - Ý nghĩa: mask loss ép mặt nạ gần mặt nạ lý tưởng, SI-SNR ép dạng sóng gần mục tiêu. Hai thành phần cùng chiều (đều là mất mát).
    """)
    return


@app.cell
def _(ROOT, mo):
    import yaml

    from src.train import FRCRNTrainConfig

    _raw = yaml.safe_load((ROOT / "config" / "frcrn.yaml").read_text(encoding="utf-8"))
    train_config = FRCRNTrainConfig.from_yaml(ROOT / "config" / "frcrn.yaml")
    _rows = "\n".join(
        f"| `{k}` | `{getattr(train_config, k)}` | {'yaml' if k in _raw.get('train', {}) else 'mặc định trong code'} |"
        for k in (
            "epochs",
            "patience",
            "lr",
            "grad_clip",
            "batch_size",
            "eval_batch_size",
            "halve_lr_after",
            "segment_seconds",
            "eval_segment_seconds",
            "snr_levels",
            "lr_decay",
            "selection_metric",
            "high_snr_levels",
        )
    )
    mo.md(
        f"""
        ## 5. Vòng huấn luyện và cấu hình

        Cấu hình **đọc ngay từ** `config/frcrn.yaml` của máy này (`FRCRNTrainConfig.from_yaml`).
        Lưu ý: bản trên molab có thể khác (đã thấy: molab `epochs=100, batch=12`, local khác). Hãy so lại trước khi kết luận về một lần chạy cụ thể.

        | Tham số | Giá trị | Nguồn |
        | --- | --- | --- |
        {_rows}

        Mô hình: `target = {_raw['model']['target']}`, loss `{_raw['loss']}`.

        **Vòng lặp** (`Trainer.fit` + `FRCRNTrainer`) **[đọc]**:

        1. Đánh giá bản pretrain trên validation làm mốc "epoch 0" (`baseline`), lưu như best đầu tiên.
        2. Mỗi epoch: duyệt train; mỗi batch tính `loss`, bỏ qua nếu không hữu hạn, `backward`, cắt gradient (`grad_clip`), `Adam.step`.
        3. Đánh giá validation: `loss`, `noise_si_snr` (SI-SNR giữa nhiễu thật và `mixture − ŝ`), `speech_si_snr`, kèm trung bình ở SNR cao (`*_hi`: 15 và 20 dB).
        4. Nếu `selection_metric` tốt hơn best: lưu `best.pt` (chỉ trọng số mạng + `epoch` + giá trị chỉ số), gọi `on_checkpoint` (đẩy Drive); đặt `stale = 0`.
        5. Ngược lại `stale += 1`. Khi `stale == halve_lr_after`: **nạp lại trọng số best, giảm lr một nửa**. Khi `stale >= patience`: dừng.
        6. Kết thúc: `restore_best()` rồi `eval()`; sau đó `freeze()` để làm teacher.

        Điểm cần nhớ: `best.pt` **không chứa optimizer**, nên không resume được từ file đó; resume dùng `resume.pt` riêng (có trong `finetune_frcrn`, **không** có trong notebook `beat-v1`).
        """
    )
    return (train_config,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Kiểm tra số học (tuỳ chọn)

    Đặt `RUN_CHECKS = True` ở ô bên dưới để chạy. Việc chạy gồm: dựng `FRCRNSeparator` trên CPU, đếm tham số, một lượt xuôi
    với 1 giây tín hiệu ngẫu nhiên, một lần tính loss, và nạp thử `pretrained/FRCRN/best.pt` nếu có.
    Không đọc dữ liệu, không huấn luyện, không ghi tệp.
    """)
    return


@app.cell
def _():
    RUN_CHECKS = False
    return (RUN_CHECKS,)


@app.cell
def _(RUN_CHECKS, ROOT, mo):
    mo.stop(not RUN_CHECKS, mo.md("`RUN_CHECKS = False`: bỏ qua. Đặt `True` để chạy kiểm tra."))
    import torch

    from src.model.frcrn import FRCRNSeparator

    torch.manual_seed(0)
    separator = FRCRNSeparator.from_yaml(ROOT / "config" / "frcrn.yaml").eval()
    network = separator.network
    trainable = sum(p.numel() for p in network.parameters())
    buffers = sum(b.numel() for b in network.buffers() if b.dtype.is_floating_point)
    per_module = {
        name: sum(p.numel() for p in child.parameters()) for name, child in network.named_children()
    }

    waveform = torch.randn(2, 16000) * 0.1
    noise = torch.randn(2, 16000) * 0.05
    with torch.no_grad():
        out = network(waveform)
        estimate = separator(waveform)
        loss_value = separator.loss(waveform, noise).item()

    lines = [
        f"tham số học được (network.parameters): {trainable:,}",
        f"buffer float (STFT/iSTFT, không học): {buffers:,}",
        f"theo module: {per_module}",
        f"network(waveform).estimate: {tuple(out.estimate.shape)}",
        f"network(waveform).mask: {tuple(out.mask.shape)}",
        f"separator(waveform): {tuple(estimate.shape)} (target={separator.config.target})",
        f"loss trên dữ liệu ngẫu nhiên: {loss_value:.3f} (chỉ kiểm tra chạy được, không có ý nghĩa)",
    ]
    ckpt = ROOT / "pretrained" / "FRCRN" / "best.pt"
    if ckpt.is_file():
        report = separator.load_pretrained(ckpt)
        lines.append(f"nạp {ckpt.name}: missing={len(report.missing)}, unexpected={len(report.unexpected)}")
    else:
        lines.append(f"không có {ckpt}")
    mo.md("```\n" + "\n".join(lines) + "\n```")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Đã và chưa kiểm chứng

    **Đã kiểm chứng từ mã và tệp trong repo [đọc]**
    - Cấu trúc hai U-Net xếp tầng, mask = tanh(U1) + tanh(U2(U1)), áp mask phức, iSTFT bằng conv.
    - `best.pt` (`pretrained/FRCRN/best.pt`) là dict gồm `model` (868 tensor, 14 387 220 phần tử gồm cả buffer STFT/iSTFT), `epoch = 25`, `noise_si_snr ≈ 7,07`.
      Ba file `FRCRN/best.pt`, `frcrn_best.pt`, `frcrn_beat_v1_best.pt` giống hệt nhau (cùng MD5).
    - Cách trộn dữ liệu, tách validation, loại nhiễu held-out, loss, lịch giảm lr, tiêu chí dừng.

    **Chưa kiểm chứng**
    - `best.pt` này có đúng do cấu hình ở mục 5 tạo ra hay không. Chưa tìm thấy log chạy ngày 3/10 trong `output/log`; chỉ có chứng cứ gián tiếp
      (có `epoch` và `noise_si_snr`, ngày sửa trước commit `2ccdb45`, hợp với `epochs=100, patience=15` của notebook).
    - Hành vi bên trong `mix_at_snr` (chuẩn hoá khi trộn).
    - Cấu hình thực sự dùng trên molab ở lần chạy gốc (file trên molab hiện tại có thể đã thay đổi).

    **Cách xác nhận tiếp**: tìm `result.json` hoặc log cùng thư mục `FA26_Capstone/Truong/FRCRN` trên Drive; đối chiếu `history` trong đó
    với `epoch = 25` và `noise_si_snr` của `best.pt`.
    """)
    return


if __name__ == "__main__":
    app.run()
