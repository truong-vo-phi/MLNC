import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    mo.output.clear_console()
    return


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import torch
    import yaml

    ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Gốc dữ liệu: biến môi trường ưu tiên hơn config/url.yaml (url.yaml trỏ tới D:\content trên máy local).
    # Chỉ đặt khi thư mục có thật.
    for _env, _base, _marker in (
        ("MLNC_MIXTURE_36L", "/marimo/datasets/mix-dataset", "labels.txt"),
        ("MLNC_AUDIOSET_40L", "/marimo/datasets/audioset", "metadata/manifest.csv"),
    ):
        # Kaggle giải nén lồng một cấp: gốc thật là thư mục chứa tệp đánh dấu (giống find_root trong scripts/download_dataset.py)
        _tail = Path(_marker).parts
        _hits = sorted(Path(_base).rglob(_tail[-1]), key=lambda p: len(p.parts)) if Path(_base).is_dir() else []
        _hits = [p for p in _hits if p.parts[-len(_tail) :] == _tail]
        if _hits:
            os.environ.setdefault(_env, str(_hits[0].parents[len(_tail) - 1]))

    RAW_CONFIG = yaml.safe_load((ROOT / "config" / "frcrn.yaml").read_text(encoding="utf-8"))
    return Path, RAW_CONFIG, ROOT, json, torch


@app.cell
def _(mo):
    import importlib.util

    REQUIRED_MODULES = ("soundfile", "torchaudio", "timm", "scipy")
    MISSING_MODULES = [_name for _name in REQUIRED_MODULES if importlib.util.find_spec(_name) is None]
    mo.callout(
        mo.md(
            f"Thiếu module: `{', '.join(MISSING_MODULES)}`. `scripts.load_dataset` và `src.train` chưa import được, nên các ô cấu hình và chạy bị chặn. "
            "Cài trong môi trường rồi khởi động lại kernel (không cài giữa phiên: cài `timm` từng nâng `torch` dưới kernel đang chạy)."
        )
        if MISSING_MODULES
        else mo.md("Đủ module cần cho `scripts.sweep_frcrn`."),
        kind="danger" if MISSING_MODULES else "success",
    )
    return (MISSING_MODULES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # frcrn-verify: fine-tune FRCRN trên mix-dataset với target noise

    Notebook này **mô tả, kiểm tra và (khi bạn bật cờ) chạy** việc fine-tune FRCRN trên mix-dataset, với `target = noise`.
    Mọi thứ (giải thích, sơ đồ, cấu hình, ô chạy, ô đọc kết quả) nằm trong chính file này.

    **Bài toán.** Dùng mask loại tiếng nói khỏi mixture để lấy nhiễu. Đầu vào mạng là `mixture`, tham chiếu của loss là `noise`.

    **Mặc định không chạy gì nặng.** Mọi cờ `RUN_*` đều là `False`. *Run all* chỉ dựng cấu hình và hiển thị.

    Nhãn độ chắc chắn: **[đọc]** lấy từ mã hoặc tệp trong repo, **[suy luận]** suy ra từ mã, **[chưa kiểm chứng]** chưa có bằng chứng.

    | Mục | Nội dung | Cờ |
    | --- | --- | --- |
    | 1 | Pipeline fine-tune | |
    | 2 | Dữ liệu: mix-dataset | `RUN_DATA_CHECK` |
    | 3 | Kiến trúc FRCRN | |
    | 4 | `target = noise`: mask lý tưởng | |
    | 5 | Hàm loss | `RUN_CHECKS` |
    | 6 | Cấu hình fine-tune | |
    | 7 | Chạy fine-tune | `RUN_FINETUNE` |
    | 8 | Đọc kết quả | |
    | 9 | Kiểm duyệt pipeline run | |
    | 10 | Checklist trước khi chạy trên molab | |
    | 11 | Đã và chưa kiểm chứng | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pipeline fine-tune

    Khởi tạo từ checkpoint pretrain FRCRN_SE_16K, huấn luyện trên mix-dataset (mixture đã trộn sẵn), tham chiếu là nhiễu.
    Chỉ số chọn best là `noise_si_snr_hi` (trung bình SI-SNR của nhiễu ở 15 và 20 dB, trên validation). **[đọc]** `trial_train_config` trong `scripts/sweep_frcrn.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 680 510' width='680' height='510' role='img' style='max-width:100%;height:auto'><rect width='680' height='510' fill='white'/><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker></defs><rect x='240' y='20' width='200' height='56' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='44.333333333333336' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mix-dataset</text><text x='340.0' y='60.333333333333336' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>một hàng = một mẫu</text><rect x='40' y='110' width='180' height='56' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='130.0' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mixture_path</text><text x='130.0' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>đầu vào mạng</text><rect x='250' y='110' width='180' height='56' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>noise_path</text><text x='340.0' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tham chiếu, đã co theo SNR</text><rect x='460' y='110' width='180' height='56' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4' stroke-dasharray='5 4'/><text x='550.0' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>clean_path</text><text x='550.0' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>không dùng ở target noise</text><path d='M300,76 L300,92 L130,92 L130,110' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M340,76 L340,110' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M380,76 L380,92 L550,92 L550,110' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><rect x='40' y='210' width='180' height='70' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='130.0' y='241.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FRCRN, target noise</text><text x='130.0' y='257.33333333333337' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>khởi tạo FRCRN_SE_16K</text><rect x='250' y='210' width='180' height='70' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='241.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Loss</text><text x='340.0' y='257.33333333333337' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mask loss - SI-SNR</text><rect x='460' y='210' width='180' height='70' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='550.0' y='241.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Adam.step</text><text x='550.0' y='257.33333333333337' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>cập nhật FRCRN</text><path d='M130,166 L130,210' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M340,166 L340,210' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='350' y='192' text-anchor='start' font-size='11.5' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>noise</text><path d='M220,245 L250,245' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='235' y='238' text-anchor='middle' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>n̂</text><path d='M430,245 L460,245' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='460' y='330' width='180' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='550.0' y='354.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Validation</text><text x='550.0' y='370.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>noise_si_snr mỗi epoch</text><path d='M550,280 L550,330' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M550,386 L550,408' fill='none' stroke='#334155' stroke-width='1.4'/><path d='M130,408 L550,408' fill='none' stroke='#334155' stroke-width='1.4'/><rect x='40' y='430' width='180' height='56' rx='8' fill='#fce7f3' stroke='#334155' stroke-width='1.4'/><text x='130.0' y='454.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>best.pt</text><text x='130.0' y='470.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tốt hơn best: lưu</text><rect x='250' y='430' width='180' height='56' rx='8' fill='#fee2e2' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='454.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Stale 3</text><text x='340.0' y='470.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>nạp lại best, lr x 0,5</text><rect x='460' y='430' width='180' height='56' rx='8' fill='#fee2e2' stroke='#334155' stroke-width='1.4'/><text x='550.0' y='454.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Stale 15</text><text x='550.0' y='470.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>dừng sớm</text><path d='M130,408 L130,430' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M340,408 L340,430' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M550,408 L550,430' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/></svg>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dữ liệu: mix-dataset

    Dùng lớp `Mixture36Dataset` (`scripts/load_dataset.py`) qua `finetune_frcrn` (`scripts/finetune.py`). **[đọc]**

    | Thành phần | Nguồn | Vai trò |
    | --- | --- | --- |
    | `waveform` | cột `mixture_path` | đầu vào của mạng |
    | `noise` | cột `noise_path` (`load_noise=True`) | tham chiếu của loss khi `target = noise`. **Đã được nhân theo SNR** khi tạo mix-dataset (theo bạn), nên là thành phần nhiễu thật nằm trong mixture |
    | `clean` | cột `clean_path` (`load_clean=True`) | không dùng ở `target = noise` |
    | `snr_db` | cột `target_snr_db` | thống kê `noise_si_snr` theo từng mức SNR, không đi vào loss |

    - Split theo doc: train 30.240, validation 6.480, test 6.480 mẫu; SNR −5, 0, 5, 10, 15, 20 dB. Giai đoạn fine-tune chỉ dùng train và validation, **không đụng test**. **[đọc]**
    - Cắt đoạn: train 2 s, validation 4 s (`segment_seconds`, `eval_segment_seconds`). **[đọc]**
    - Khác với `DynamicMixDataset` (trộn động tiếng nói với nhiễu AudioSet): ở đây mixture là cố định, 6 mức SNR cố định, nên phân phối khớp chính xác validation và test nhưng ít đa dạng hơn. **[suy luận]**
    - Với `target = speech`, trainer tính tiếng nói là `mixture − noise`. Với `target = noise` thì **không cần** phép trừ này: tham chiếu là chính `noise_path`. **[đọc]** `training_target` trong `separator.py`.
    - Công thức nhân SNR khi tạo mix-dataset **không nằm trong repo** (bộ nạp chỉ đọc file). Nếu cùng công thức với `mix_at_snr` của `DynamicMixDataset` (co nhiễu để công suất nhiễu thấp hơn công suất tiếng nói đúng `snr_db` dB, rồi chia cả mixture và nhiễu cho đỉnh biên độ nếu đỉnh vượt 1) thì `mixture = clean + noise` đúng tới hệ số đỉnh chung. **[chưa kiểm chứng]**; kiểm bằng ô `RUN_DATA_CHECK` bên dưới (so SNR đo từ `clean`/`noise` với nhãn `target_snr_db`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 680 440' width='680' height='440' role='img' style='max-width:100%;height:auto'><rect width='680' height='440' fill='white'/><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker></defs><rect x='190' y='20' width='300' height='56' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='44.333333333333336' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mix-dataset, theo split</text><text x='340.0' y='60.333333333333336' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>train 30.240, val 6.480, test 6.480</text><rect x='40' y='110' width='135' height='56' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='107.5' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mixture_path</text><text x='107.5' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>đầu vào</text><path d='M107,166 L107,210' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='195' y='110' width='135' height='56' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='262.5' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>noise_path</text><text x='262.5' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>đã co theo SNR</text><path d='M262,166 L262,210' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='350' y='110' width='135' height='56' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4' stroke-dasharray='5 4'/><text x='417.5' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>clean_path</text><text x='417.5' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tuỳ chọn</text><path d='M417,166 L417,210' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='505' y='110' width='135' height='56' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4'/><text x='572.5' y='134.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>target_snr_db</text><text x='572.5' y='150.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mức SNR</text><path d='M572,166 L572,210' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M250,76 L250,92 L107,92 L107,110' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M310,76 L310,92 L262,92 L262,110' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M370,76 L370,92 L417,92 L417,110' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M430,76 L430,92 L572,92 L572,110' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='210' width='600' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='234.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Mixture36Dataset</text><text x='340.0' y='250.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>load_noise=True, load_clean tuỳ chọn</text><path d='M340,266 L340,300' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='190' y='300' width='300' height='56' rx='8' fill='#fce7f3' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='324.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Batch</text><text x='340.0' y='340.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>waveform (mixture), noise, snr_db</text><text x='40' y='392' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>Đoạn cắt: train 2 s, validation 4 s (cùng một lời gọi đọc cho cả ba file).</text><text x='40' y='412' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>noise_path đã nhân theo SNR (theo bạn); mixture = clean + noise: kiểm bằng RUN_DATA_CHECK.</text></svg>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nguồn dữ liệu hiện dùng: speech mix-dataset + nhiễu AudioSet 36L (trộn động)

    Thay vì khôi phục `noise_path`, lần chạy dùng `clean_path` (tiếng nói) của mix-dataset trộn động bằng `mix_at_snr` với clip AudioSet 40L **đã loại 4 nhãn
    không thuộc 36 lớp** (Bass drum, Snare drum, Cello, Double bass; clip mang bất kỳ nhãn nào trong 4 nhãn này bị bỏ). `mixture = speech + noise` chính xác nên
    tham chiếu nhiễu đúng thang, không cần bảng offset. **[đọc]** `scripts/dynamic_mix_36l.py`. Phần dưới là ghi chép hướng khôi phục cũ (hướng A), không còn nằm trên đường chạy.

    ### Thử khôi phục nhiễu tham chiếu từ AudioSet (hướng A)

    Yêu cầu: tham chiếu nhiễu lấy từ clip AudioSet gốc, không lấy từ `noise_path`. Đã thử khôi phục `noise_path` từ AudioSet trên 15 hàng validation
    (11 hàng thường và 4 hàng `noise_scale` 125–461): clip gốc 44,1 kHz int16, đổi mono, resample 16 kHz, tìm đoạn 4 s khớp nhất bằng tương quan chéo. **[đo]**

    | Kết quả | Chi tiết |
    | --- | --- |
    | Hệ số nhân | hệ số khớp ≈ `noise_scale × post_gain`, lệch ≤ ~1% ở cả 15 hàng (kể cả hàng cực đoan) |
    | Dạng sóng | sai số năng lượng tương đối: tốt nhất 8e-6 đến 2e-5 (2 hàng); điển hình 7e-4 đến 2e-2 (7 hàng); kém 5% đến 24% (6 hàng, gồm cả 4 hàng cực đoan) |
    | Kênh và resample | trung bình hai kênh tốt hơn kênh trái hoặc phải; `resample_poly`, FFT, torchaudio, soxr cho sai số gần như nhau |
    | Đoạn cắt | không có quy tắc đơn giản (0,0 s, 6,0 s và các giá trị ngẫu nhiên); cắt ở miền 44,1 kHz rồi resample không cải thiện; `random_seed` của hàng không tái lập được offset bằng 4 cách rút thông thường (`default_rng.integers`, `RandomState.randint`, `random.Random.randint`, `default_rng.random × slack`) trên 8 hàng |

    Kết luận: **chưa khôi phục chính xác được** nhiễu từ AudioSet; script sinh mix-dataset không có trong repo. Sai số tương đối ε giới hạn SI-SNR đo được ở khoảng `−10·log10(ε)` dB
    (ε = 5% cho ≈ 13 dB, ε = 24% cho ≈ 6 dB). Vì vậy các ô bên dưới **vẫn đọc nhiễu từ `noise_path`**; hướng khôi phục từ AudioSet chưa được đưa vào ô chạy. **[suy luận từ công thức]**
    """)
    return


@app.cell
def _(mo, torch):
    RUN_DATA_CHECK = False  # True: đọc ~12 mẫu validation từ mix-dataset (cần dataset trên máy), kiểm mixture = clean + noise

    mo.stop(not RUN_DATA_CHECK, mo.md("`RUN_DATA_CHECK = False`: bỏ qua. Đặt `True` để kiểm tra dữ liệu."))

    from scripts.load_dataset import Mixture36Dataset

    try:
        _dataset = Mixture36Dataset(split="validation", duration=2.0, load_noise=True, load_clean=True)
    except Exception as _error:
        print("Không mở được mix-dataset:", repr(_error))
    else:
        print(f"validation: {len(_dataset)} mẫu")
        print("chỉ mục | SNR nhãn | SNR đo từ clean/noise | ||mixture-clean-noise||² / ||mixture||²")
        for _index in range(0, len(_dataset), max(len(_dataset) // 12, 1)):
            _item = _dataset[_index]
            _mixture, _noise, _clean = _item["waveform"], _item["noise"], _item["clean"]
            _residual = (_mixture - _clean - _noise).square().sum() / _mixture.square().sum()
            _measured = 10 * torch.log10(_clean.square().sum() / _noise.square().sum())
            print(f"{_index:7d} | {float(_item['snr_db']):8.1f} | {float(_measured):21.2f} | {float(_residual):.2e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Kiến trúc FRCRN

    Lớp bọc `FRCRNSeparator` chuẩn hoá biên độ rồi gọi mạng `FRCRN` (`src/model/frcrn/`). **[đọc]**

    1. `audio_norm`: đưa mixture về mức −25 dB, giữ lại hệ số `scale`.
    2. `ConvSTFT`: cửa sổ 640, hop 320, 321 bin; tách thực và ảo.
    3. `UNet` 1 nhận `[B, 1, 321, T, 2]` rồi trả mask thô; `UNet` 2 nhận đầu ra của `UNet` 1.
    4. **Mask cuối = tanh(U1) + tanh(U2(U1))**, giá trị trong (−2, 2).
    5. Nhân phức mask với phổ vào, `ConviSTFT` + overlap-add, cắt về độ dài gốc, nhân lại `scale`.
    6. Với `target = noise`, `forward` trả **thẳng** đầu ra này (n̂), không trừ khỏi mixture.

    Mạng giống hệt khi `target = speech`; chỉ khác ý nghĩa của đầu ra và mục tiêu huấn luyện.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 780 880' width='780' height='880' role='img' style='max-width:100%;height:auto'><rect width='780' height='880' fill='white'/><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker></defs><rect x='250' y='20' width='280' height='44' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='46.333333333333336' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>waveform (mixture)</text><rect x='250' y='88' width='280' height='44' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='114.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>audio_norm: mức -25 dB, giữ scale</text><rect x='250' y='156' width='280' height='44' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='174.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>ConvSTFT: win 640, hop 320</text><text x='390.0' y='190.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>321 bin</text><rect x='250' y='224' width='280' height='44' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='242.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tách thực / ảo</text><text x='390.0' y='258.33333333333337' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>[B, 1, 321, T, 2]</text><rect x='250' y='292' width='280' height='44' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='310.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>UNet 1</text><text x='390.0' y='326.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>7 encoder + FSMN + SE + 7 decoder</text><rect x='250' y='496' width='280' height='44' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='514.3333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>cộng: mask phức</text><text x='390.0' y='530.3333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tanh(U1) + tanh(U2(U1))</text><rect x='250' y='564' width='280' height='44' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='590.3333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>nhân phức: mask x phổ vào</text><rect x='250' y='632' width='280' height='44' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='658.3333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>ConviSTFT + overlap-add</text><rect x='250' y='700' width='280' height='44' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='726.3333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>cắt về độ dài gốc, nhân scale</text><rect x='250' y='768' width='280' height='52' rx='8' fill='#fce7f3' stroke='#334155' stroke-width='1.4'/><text x='390.0' y='790.3333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>n̂ (target = noise)</text><text x='390.0' y='806.3333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>forward trả thẳng n̂, không trừ</text><path d='M390.0,64 L390.0,88' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,132 L390.0,156' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,200 L390.0,224' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,268 L390.0,292' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,540 L390.0,564' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,608 L390.0,632' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,676 L390.0,700' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M390.0,744 L390.0,768' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='370' width='150' height='44' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4'/><text x='115.0' y='396.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tanh</text><rect x='580' y='355' width='170' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='665.0' y='379.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>UNet 2</text><text x='665.0' y='395.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>(cùng cấu trúc)</text><rect x='580' y='438' width='170' height='44' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4'/><text x='665.0' y='464.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>tanh</text><path d='M310,336 L310,392 L190,392' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M490,336 L490,383 L580,383' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='470' y='375' text-anchor='middle' font-size='11.5' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>đầu ra UNet 1</text><path d='M665,411 L665,438' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M115,414 L115,518 L250,518' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M665,482 L665,518 L530,518' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M250,246 L14,246 L14,586 L250,586' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='24' y='238' text-anchor='start' font-size='11.5' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>phổ vào (Re, Im)</text></svg>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1010 470' width='1010' height='470' role='img' style='max-width:100%;height:auto'><rect width='1010' height='470' fill='white'/><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker></defs><text x='505' y='28' text-anchor='middle' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Một U-Net (encoder → cổ chai → decoder; nét đứt là skip)</text><rect x='30' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='90.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Enc 1</text><text x='90.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>1 → 128</text><rect x='170' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='230.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN + Enc 2</text><text x='230.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>128</text><path d='M150,83 L170,83' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='310' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='370.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN + Enc 3</text><text x='370.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>128</text><path d='M290,83 L310,83' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='450' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='510.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN + Enc 4</text><text x='510.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>128</text><path d='M430,83 L450,83' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='590' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='650.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN + Enc 5</text><text x='650.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>128</text><path d='M570,83 L590,83' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='730' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='790.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN + Enc 6</text><text x='790.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>128</text><path d='M710,83 L730,83' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='870' y='50' width='120' height='66' rx='8' fill='#dbeafe' stroke='#334155' stroke-width='1.4'/><text x='930.0' y='79.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN + Enc 7</text><text x='930.0' y='95.33333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>128</text><path d='M850,83 L870,83' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='870' y='160' width='120' height='50' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='930.0' y='181.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>FSMN cổ chai</text><text x='930.0' y='197.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>(L1, L2)</text><path d='M930.0,116 L930.0,160' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='870' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='930.0' y='299.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 1</text><text x='930.0' y='315.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>+ FSMN</text><rect x='730' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='790.0' y='299.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 2</text><text x='790.0' y='315.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>+ FSMN</text><path d='M870,303 L850,303' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='590' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='650.0' y='299.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 3</text><text x='650.0' y='315.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>+ FSMN</text><path d='M730,303 L710,303' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='450' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='510.0' y='299.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 4</text><text x='510.0' y='315.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>+ FSMN</text><path d='M590,303 L570,303' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='310' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='370.0' y='299.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 5</text><text x='370.0' y='315.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>+ FSMN</text><path d='M450,303 L430,303' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='170' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='230.0' y='299.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 6</text><text x='230.0' y='315.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>+ FSMN</text><path d='M310,303 L290,303' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='30' y='270' width='120' height='66' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='90.0' y='307.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Dec 7</text><path d='M170,303 L150,303' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M930.0,210 L930.0,270' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M90.0,116 L90.0,270' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><path d='M230.0,116 L230.0,270' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><path d='M370.0,116 L370.0,270' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><path d='M510.0,116 L510.0,270' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><path d='M650.0,116 L650.0,270' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><path d='M790.0,116 L790.0,270' fill='none' stroke='#334155' stroke-width='1.4' stroke-dasharray='6 4' marker-end='url(#a)'/><rect x='30' y='390' width='120' height='50' rx='8' fill='#fce7f3' stroke='#334155' stroke-width='1.4'/><text x='90.0' y='411.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Conv 1x1</text><text x='90.0' y='427.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>→ mask (Re, Im)</text><path d='M90.0,336 L90.0,390' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='190' y='402' text-anchor='start' font-size='11.5' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>Nét đứt (skip): đầu ra encoder k nối vào đầu vào Dec (8-k).</text><text x='190' y='424' text-anchor='start' font-size='11.5' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>Kernel encoder (5,2) x6 rồi (2,2), stride (2,1); SE sau mỗi encoder và sau Dec 1…5.</text><text x='190' y='438' text-anchor='start' font-size='11.5' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>Mọi phép (conv, BN, FSMN, SE) có hai nhánh thực/ảo, kết hợp theo phép nhân phức.</text></svg>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. `target = noise`: mask lý tưởng

    Với phổ phức của mixture `Y = S + N`:

    | | `target = speech` | `target = noise` |
    | --- | --- | --- |
    | Mạng học | `M̂·Y ≈ S` | `M̂·Y ≈ N` |
    | Mask lý tưởng | `S / Y` | `N / Y = 1 − S/Y` |
    | `estimate` của mạng | tiếng nói ŝ | nhiễu n̂ |
    | `forward` trả về | `mixture − ŝ` | `n̂` |
    | Tham chiếu của loss | `mixture − noise` (đã chuẩn hoá) | `noise / scale` |

    Ví dụ một ô: `S = 0,6+0,8j`, `N = 0,2−0,1j` nên `Y = 0,8+0,7j`. Khi đó mask tiếng nói `= 0,920+0,195j`, mask nhiễu `= 0,080−0,195j`, tổng bằng 1.

    Ở SNR cao nhiễu yếu nên mask nhiễu nhỏ: ví dụ `|S| ≫ |N|` cho mask nhiễu ≈ `0,039−0,018j`; ở SNR thấp (`|S| ≪ |N|`) mask nhiễu ≈ `0,529−0,118j`. **[tính tay]**

    Lưu ý khi chuyển từ `speech` sang `noise` **[suy luận]**:
    - Checkpoint pretrain được huấn luyện cho tách tiếng nói, nên mask ban đầu gần `S/Y`, còn mục tiêu mới là `1 − S/Y`: mạng phải học lại. Điểm epoch 0 (baseline) sẽ thấp.
    - Bằng chứng sweep về `noise` còn yếu: hai lượt chỉ chạy 2 epoch rồi bị dừng sớm (−3,67 và −6,21 dB), khác cấu hình nhau, nên chưa kết luận được. **[đọc, theo doc]**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Hàm loss

    `FRCRNLoss` (`src/loss/frcrn.py`). **[đọc]** Gọi `normalized` là mixture đã chuẩn hoá, `n` là nhiễu chuẩn hoá `noise / scale`, `n̂` là đầu ra của mạng:

    $$\mathcal{L}= w_m\cdot 640\Big[\mathrm{MSE}(\hat M_{re},M^*_{re})+\mathrm{MSE}(\hat M_{im},M^*_{im})\Big]\;-\;\overline{\mathrm{SI\text{-}SNR}(n,\hat n)},\qquad M^*=\frac{N\,\overline{Y}}{|Y|^2+\varepsilon}$$

    - Mask lý tưởng cắt giá trị ngoài ±2 (`mask_bound`) về ±1. `mask_weight` $w_m$ mặc định 1,0.
    - MSE là trung bình trên toàn bộ `B × 321 × T` phần tử, tính riêng cho nửa thực và nửa ảo rồi cộng.
    - **SI-SNR**: trừ trung bình, chiếu `n̂` lên `n` ($\alpha = \langle \hat n, n\rangle / \lVert n\rVert^2$), phần dư $e=\hat n-\alpha n$, rồi $10\log_{10}\big(\lVert \alpha n\rVert^2 / \lVert e\rVert^2\big)$, thêm `eps = 1e-6`.
    - Bất biến với hệ số khuếch đại, đảo dấu và lệch DC. Không bị chặn trên (tối đa ≈ 96 dB ở `eps = 1e-6` với đoạn 4.000 mẫu). **[chạy thử]**
    - Thang của hai thành phần: nếu mỗi MSE bằng 0,01 thì mask loss ≈ 12,8 (với `mask_weight = 1`), có thể lớn hơn nhiều so với SI-SNR (cỡ chục dB). Trong sweep, `loss` trên W&B ≈ 570–616 khi `mask_weight = 4`, tức mask loss chiếm gần hết giá trị. **[suy luận từ số liệu W&B]**
    - Loss không phụ thuộc `snr_db`; nhãn SNR chỉ dùng để thống kê lúc đánh giá.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 680 580' width='680' height='580' role='img' style='max-width:100%;height:auto'><rect width='680' height='580' fill='white'/><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker></defs><rect x='240' y='20' width='200' height='56' rx='8' fill='#f1f5f9' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='44.333333333333336' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Batch</text><text x='340.0' y='60.333333333333336' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mixture, noise</text><rect x='240' y='106' width='200' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='340.0' y='130.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>audio_norm</text><text x='340.0' y='146.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mức -25 dB, giữ scale</text><rect x='30' y='196' width='190' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='125.0' y='220.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Target nhiễu</text><text x='125.0' y='236.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>noise / scale</text><rect x='460' y='196' width='190' height='56' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='555.0' y='220.33333333333334' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Mạng FRCRN</text><text x='555.0' y='236.33333333333334' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>ra mask và n̂</text><rect x='30' y='286' width='190' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='125.0' y='310.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Mask lý tưởng</text><text x='125.0' y='326.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>cIRM N/Y từ target</text><rect x='60' y='376' width='200' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='160.0' y='400.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Mask loss</text><text x='160.0' y='416.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>MSE thực + ảo, x 640</text><rect x='360' y='456' width='200' height='56' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.4'/><text x='460.0' y='480.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>SI-SNR</text><text x='460.0' y='496.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>trung bình theo batch</text><rect x='60' y='456' width='200' height='56' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.4'/><text x='160.0' y='480.3333333333333' text-anchor='middle' font-size='13' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Loss tổng</text><text x='160.0' y='496.3333333333333' text-anchor='middle' font-size='13' font-weight='normal' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>mask loss - SI-SNR</text><path d='M340,76 L340,106' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M240,48 L60,48 L60,196' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='68' y='70' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>noise</text><path d='M240,134 L125,134 L125,196' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M440,134 L555,134 L555,196' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M125,252 L125,286' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M125,342 L125,376' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M500,252 L500,404 L260,404' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='508' y='330' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>mask</text><path d='M610,252 L610,484 L560,484' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='618' y='370' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>n̂</text><path d='M30,224 L14,224 L14,542 L460,542 L460,512' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='237' y='534' text-anchor='middle' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>target</text><path d='M160,432 L160,456' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><path d='M360,484 L260,484' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><text x='310' y='476' text-anchor='middle' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>trừ</text></svg>""")
    return


@app.cell
def _(RAW_CONFIG, mo, torch):
    RUN_CHECKS = False  # True: kiểm tra số học trên CPU với dữ liệu ngẫu nhiên (nhẹ, không đọc dữ liệu thật, không ghi tệp)

    mo.stop(not RUN_CHECKS, mo.md("`RUN_CHECKS = False`: bỏ qua. Đặt `True` để chạy kiểm tra."))

    from src.loss.frcrn import FRCRNLoss, complex_ideal_ratio_mask, si_snr
    from src.model.frcrn import FRCRNSeparator
    from src.model.frcrn.config import FRCRNConfig

    _config = FRCRNConfig.from_dict({**RAW_CONFIG[FRCRNConfig.section], "target": "noise"})
    _separator = FRCRNSeparator(_config, FRCRNLoss(_config, **RAW_CONFIG.get("loss", {}))).eval()

    torch.manual_seed(0)
    _speech, _noise = torch.randn(2, 16000) * 0.1, torch.randn(2, 16000) * 0.03
    _mixture = _speech + _noise

    print("tham số học được:", f"{sum(p.numel() for p in _separator.network.parameters()):,}")
    print("training_target(noise) == noise:", bool(torch.equal(_separator.training_target(_mixture, _noise), _noise)))
    with torch.no_grad():
        _estimate = _separator(_mixture)
        _loss_value = _separator.loss(_mixture, _noise).item()
    print("forward:", tuple(_estimate.shape), "| loss trên dữ liệu ngẫu nhiên:", round(_loss_value, 3), "(chỉ kiểm tra chạy được)")

    _bins = _config.num_bins
    _total = complex_ideal_ratio_mask(_noise, _mixture, _config, mask_bound=1e6) + complex_ideal_ratio_mask(
        _speech, _mixture, _config, mask_bound=1e6
    )
    print("mask nhiễu + mask tiếng nói: sai lệch thực so với 1 =", float((_total[:, :_bins] - 1).abs().max()),
          "| sai lệch ảo so với 0 =", float(_total[:, _bins:].abs().max()))

    _toy = _noise + 0.1 * torch.randn_like(_noise)
    for _name, _value in (("est", _toy), ("3 * est", 3 * _toy), ("-est", -_toy)):
        print(f"SI-SNR(noise, {_name}) = {si_snr(_noise, _value).mean().item():.3f} dB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Cấu hình fine-tune

    Cấu hình đọc từ `config/frcrn.yaml` của máy này, rồi áp `SETTINGS` bằng `trial_train_config` (chọn best theo `noise_si_snr_hi`). **[đọc]**

    Hai lựa chọn nền (`BASE_CHOICE`):
    - `"yaml"`: giữ giá trị trong `config/frcrn.yaml` (mặc định của repo), chỉ đổi `target = noise`.
    - `"sweep_best"`: bộ tốt nhất của sweep W&B (lượt `xqrcbdzg`). Bộ này được dò cho `target = speech`, nên **thiên lợi cho `speech`**; dùng cho `noise` là điểm xuất phát chứ chưa phải tối ưu. **[đọc, theo doc]**

    Cấu hình trên molab khác bản local (đã thấy: molab `epochs 100, batch 12`). Hãy đối chiếu trước khi chạy.
    """)
    return


@app.cell
def _(MISSING_MODULES, RAW_CONFIG, ROOT, mo):
    mo.stop(
        bool(MISSING_MODULES),
        mo.md(f"Thiếu module `{', '.join(MISSING_MODULES)}`: chưa import được `scripts.sweep_frcrn`. Xem ô kiểm tra phụ thuộc phía trên."),
    )

    from scripts.sweep_frcrn import ORIGINAL_CHECKPOINT, best_epoch_scores, run_trial, trial_train_config
    from src.train import FRCRNTrainConfig

    BASE_CHOICE = "yaml"  # "yaml" hoặc "sweep_best"
    SWEEP_BEST = {
        "lr": 0.000273365064091293,
        "lr_decay": 1.0,
        "halve_lr_after": 2,
        "batch_size": 12,
        "segment_seconds": 2.0,
        "grad_clip": 5.0,
        "mask_weight": 4.0,
    }
    SETTINGS = {"target": "noise", **(SWEEP_BEST if BASE_CHOICE == "sweep_best" else {})}

    base_config = FRCRNTrainConfig.from_yaml(ROOT / "config" / "frcrn.yaml")
    train_config = trial_train_config(base_config, SETTINGS)
    _fields = (
        "epochs", "patience", "lr", "lr_decay", "halve_lr_after", "grad_clip", "batch_size", "eval_batch_size",
        "segment_seconds", "eval_segment_seconds", "snr_levels", "selection_metric", "high_snr_levels",
    )
    _rows = "\n".join(f"| `{k}` | `{getattr(train_config, k)}` |" for k in _fields)
    mo.md(
        f"""**Nền:** `{BASE_CHOICE}` | **target:** `{SETTINGS['target']}` | **mask_weight:** `{SETTINGS.get('mask_weight', 1.0)}`

    | Tham số | Giá trị |
    | --- | --- |
    {_rows}

    Loss (từ yaml): `{RAW_CONFIG.get('loss')}` | checkpoint khởi tạo: `{ORIGINAL_CHECKPOINT}` (tồn tại: {ORIGINAL_CHECKPOINT.is_file()})"""
    )
    return (
        BASE_CHOICE,
        ORIGINAL_CHECKPOINT,
        SETTINGS,
        base_config,
        best_epoch_scores,
        run_trial,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Chạy fine-tune

    `RUN_FINETUNE = False` mặc định, và ô còn bị chặn bởi `ALLOW_NOISE_PATH_REFERENCE = False` (xem mục 9, phát hiện 1). Khi mở, ô dưới gọi `run_trial` (`scripts/sweep_frcrn.py`), tức `finetune_frcrn` trên `Mixture36Dataset` với `target = noise`. **[đọc]**

    Trước khi bật:
    - Cần có mix-dataset giải nén sẵn và checkpoint pretrain `pretrained/last_best_checkpoint.pt` (FRCRN_SE_16K gốc).
    - Thời gian mỗi epoch chưa đo trên toàn bộ dữ liệu (cấu hình sweep ghi "~30 phút/epoch, ước tính thô"); dùng `MAX_SAMPLES = 1200` để chạy thử nhanh như sweep. **[đọc]**
    - Ghi vào `output/` (sao lưu Drive nếu `USE_DRIVE = True`). Không ghi đè lần chạy cũ vì `RunPaths.new` tạo thư mục mới theo thời gian.
    """)
    return


@app.cell
def _(
    BASE_CHOICE,
    ORIGINAL_CHECKPOINT,
    ROOT,
    SETTINGS,
    base_config,
    json,
    mo,
    run_trial,
    torch,
):
    RUN_FINETUNE = False  # True: bắt đầu fine-tune (tốn GPU và thời gian)
    # Yêu cầu hiện tại: tham chiếu nhiễu lấy từ AudioSet, KHÔNG lấy từ noise_path của mix-dataset.
    # Đường chạy bên dưới (finetune_frcrn -> Mixture36Dataset) vẫn đọc noise_path, nên bị chặn mặc định.
    # Chỉ đặt True khi bạn chủ động chấp nhận dùng noise_path (trái yêu cầu trên).
    ALLOW_NOISE_PATH_REFERENCE = False
    SMOKE = True  # True: 1 epoch, 64 mẫu để kiểm đường chạy; False: chạy thật
    USE_DRIVE = False  # True: đăng nhập Google Drive, sao lưu best.pt/resume.pt/result.json
    DRIVE_DIR = "FA26_Capstone/Truong/frcrn-noise"  # thư mục Drive MỚI (không trùng thư mục có sẵn)
    MAX_SAMPLES = None  # None: toàn bộ train/validation; 1200: như sweep
    NUM_WORKERS = 4

    mo.stop(not RUN_FINETUNE, mo.md("`RUN_FINETUNE = False`: không chạy gì."))
    mo.stop(
        not ALLOW_NOISE_PATH_REFERENCE,
        mo.md(
            "**Không chạy.** Tham chiếu nhiễu hiện vẫn đọc từ `noise_path` (mix-dataset), trái yêu cầu lấy nhiễu từ AudioSet. "
            "Xem mục 9 (phát hiện 1 và 2)."
        ),
    )
    mo.stop(not ORIGINAL_CHECKPOINT.is_file(), mo.md(f"Thiếu checkpoint pretrain: `{ORIGINAL_CHECKPOINT}`"))

    from src.utils.logger import setup_logger
    from src.utils.paths import RunPaths

    _run = RunPaths.new("frcrn-noise", ROOT / "output").create()
    _logger = setup_logger(_run.run_id, _run.log_file)
    _logger.info("run %s | target=%s | base=%s | smoke=%s | max_samples=%s", _run.run_id, SETTINGS["target"], BASE_CHOICE, SMOKE, MAX_SAMPLES)

    _fs = None
    if USE_DRIVE:
        import gdrive_fsspec

        _fs = gdrive_fsspec.GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})

    _result = run_trial(
        SETTINGS,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        trial_dir=_run.checkpoint_dir,
        checkpoint=ORIGINAL_CHECKPOINT,
        base_config=base_config,
        max_samples=MAX_SAMPLES,
        num_workers=NUM_WORKERS,
        smoke=SMOKE,
        fs=_fs,
        drive_dir=DRIVE_DIR,
    )
    _history = _run.history_path("frcrn_noise")
    _history.write_text(json.dumps(_result, indent=2), encoding="utf-8")
    mo.md(
        f"best_epoch **{_result['best_epoch']}** | best_score **{_result['best_score']:.3f}** | lưu tại `{_history}`. "
        "Đặt `RESULT_JSON` ở ô kết quả bằng đường dẫn này để xem bảng."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Đọc kết quả

    Chỉ số chính: `noise_si_snr` của nhiễu thu được so với nhiễu gốc, **tách theo từng mức SNR** và `noise_si_snr_hi` (15 và 20 dB).

    Mốc tham chiếu trên **test** (6.480 mẫu), SI-SNR nhiễu, dB, theo `doc/frcrn-finetune-history.md` mục 6: **[đọc]**

    | Checkpoint | −5 | 0 | 5 | 10 | 15 | 20 | 15–20 | Trung bình |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | FRCRN_SE_16K gốc | 9,14 | 7,00 | 5,37 | 2,99 | −1,48 | −4,46 | −2,97 | 3,09 |
    | `best.pt` cũ | 13,79 | 11,06 | 8,71 | 6,43 | 3,31 | 0,60 | +1,95 | 7,32 |
    | fine-tune thẳng trên Mixture-36l (epoch 9) | 9,72 | 8,14 | 6,73 | 4,83 | 1,32 | −1,46 | −0,07 | 4,88 |

    Cách đọc thang:
    - 0 dB: năng lượng phần sai bằng phần đúng; 3 dB: sai bằng một nửa; 10 dB: sai 10%; 20 dB: sai 1%.
    - "Không lọc" (coi nguyên mixture là nhiễu) cho SI-SNR ≈ **−SNR của mixture** nếu tiếng nói và nhiễu không tương quan: +5, 0, −5, −10, −15, −20 dB. Từ đó tính SI-SNRi. **[suy luận, chưa đo]**
    - Mỗi con số chỉ 1 seed. Phải đo độ lệch chuẩn giữa các seed trước khi kết luận cách nào hơn.
    - Thước đo cuối cùng của bài toán là phân loại nhiễu (macro-F1, mAP của BEATs), không phải SI-SNR.
    """)
    return


@app.cell
def _(Path, best_epoch_scores, json, mo):
    RESULT_JSON = None  # đường dẫn tới frcrn_noise.json hoặc result.json sau khi chạy xong

    mo.stop(RESULT_JSON is None, mo.md("Chưa có kết quả: chạy mục 7 hoặc đặt `RESULT_JSON`."))

    _data = json.loads(Path(RESULT_JSON).read_text(encoding="utf-8"))
    _scores = best_epoch_scores(_data)
    _levels = (-5, 0, 5, 10, 15, 20)


    def _score(level):
        for _key in (f"noise_si_snr_{level}", f"noise_si_snr_{float(level)}"):
            if _key in _scores:
                return _scores[_key]
        return float("nan")


    _header = "| | " + " | ".join(str(v) for v in _levels) + " | 15–20 | trung bình |\n| --- |" + " --- |" * (len(_levels) + 2)
    _measured = " | ".join(f"{_score(v):.2f}" for v in _levels)
    # "không lọc" (coi mixture là nhiễu): SI-SNR ≈ -SNR; hi = trung bình 15 và 20 dB, cuối = trung bình 6 mức
    _unfiltered = " | ".join(f"{-v:+d}" for v in _levels)
    _unfiltered_hi = -sum(v for v in _levels if v in (15, 20)) / 2
    _unfiltered_mean = -sum(_levels) / len(_levels)
    mo.md(
        f"best epoch **{_data['best_epoch']}** / {_data['epochs_run']} | chọn theo `{_data['selection_metric']}` = {_data['best_score']:.3f}\n\n"
        f"{_header}\n| noise_si_snr (validation) | {_measured} | {_scores.get('noise_si_snr_hi', float('nan')):.2f} | {_scores.get('noise_si_snr', float('nan')):.2f} |\n"
        f"| không lọc (ước tính: −SNR) | {_unfiltered} | {_unfiltered_hi:.1f} | {_unfiltered_mean:.1f} |"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Kiểm duyệt pipeline run

    Rà lại đường chạy `run_trial` → `finetune_frcrn` → `FRCRNTrainer` trước khi chạy thật. Mức: **chặn** (không chạy được hoặc sai yêu cầu), **cảnh báo**, **ghi chú**.

    | # | Phát hiện | Mức | Việc cần làm |
    | --- | --- | --- | --- |
    | 1 | Tham chiếu nhiễu vẫn đọc từ `noise_path` của mix-dataset, **trái yêu cầu** lấy nhiễu từ AudioSet **[đọc]** | chặn | Ô chạy bị chặn bởi `ALLOW_NOISE_PATH_REFERENCE = False`. Cần tham chiếu từ AudioSet, hoặc bạn chủ động đặt `True` |
    | 2 | Chưa khôi phục chính xác nhiễu từ AudioSet: sai số 8e-6 đến 24% trên 15 hàng; script sinh dữ liệu không có trong repo; `random_seed` không tái lập được đoạn cắt (thử 4 cách rút thông thường trên 8 hàng) **[đo]** | chặn | Chọn: lọc hàng theo sai số, hoặc dùng `DynamicMixDataset` (nhiễu từ AudioSet, chính xác theo cấu tạo) |
    | 3 | Sandbox molab **thiếu `soundfile`, `torchaudio`, `timm`**: `scripts.load_dataset`, `src.train` không import được **[đo trên sandbox]** | chặn | Cài trong môi trường rồi khởi động lại kernel. Không cài giữa phiên: cài `timm` từng nâng `torch` dưới kernel đang chạy **[đọc doc]** |
    | 4 | `config/url.yaml` trỏ `D:\content\...`; trên molab chưa có dataset, checkpoint pretrain **[đọc]** | chặn | Đặt `MLNC_MIXTURE_36L`, `MLNC_AUDIOSET_40L` (ô setup tự đặt khi thư mục `/marimo/datasets/...` có thật); tải `pretrained/last_best_checkpoint.pt` |
    | 5 | Train luôn cắt **2 s đầu** của clip 4 s (`read_audio` với `rng=None` cho `start = 0`); validation dùng đủ 4 s. `mixture` và `noise` cắt cùng vị trí nên không lệch pha **[đọc]** | ghi chú | Nửa sau mỗi clip không bao giờ được dùng khi train. Tham chiếu từ AudioSet (nếu có) phải dùng cùng offset |
    | 6 | `USE_DRIVE = False` mà sandbox mất khi hết phiên: không có `resume.pt` **[đọc doc]** | cảnh báo | Bật Drive với thư mục mới; chạy đầu tiên dùng `SMOKE = True` (1 epoch, 64 mẫu) |
    | 7 | `src/utils/download.py` chứa token Kaggle viết thẳng, đã commit **[đọc]** | cảnh báo | Tạo lại token; file này **không** được đẩy lên molab |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(r"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 680 520' width='680' height='520' role='img' style='max-width:100%;height:auto'><rect width='680' height='520' fill='white'/><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker></defs><text x='340' y='28' text-anchor='middle' font-size='15' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>Trạng thái từng khối của pipeline run</text><rect x='40' y='50' width='600' height='58' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.2'/><rect x='40' y='50' width='14' height='58' rx='4' fill='#16a34a'/><text x='68' y='74' text-anchor='start' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>1. Dữ liệu: mix-dataset (mixture là đầu vào)</text><text x='68' y='94' text-anchor='start' font-size='12' font-weight='normal' fill='#334155' font-family='Segoe UI,Arial,sans-serif'>30.240 / 6.480 / 6.480 mẫu; các split không dùng chung tiếng nói hay nhiễu</text><path d='M340,108 L340,122' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='122' width='600' height='58' rx='8' fill='#fee2e2' stroke='#334155' stroke-width='1.2'/><rect x='40' y='122' width='14' height='58' rx='4' fill='#dc2626'/><text x='68' y='146' text-anchor='start' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>2. Tham chiếu nhiễu</text><text x='68' y='166' text-anchor='start' font-size='12' font-weight='normal' fill='#334155' font-family='Segoe UI,Arial,sans-serif'>Đang đọc noise_path (trái yêu cầu lấy từ AudioSet); khôi phục từ AudioSet chưa chính xác</text><path d='M340,180 L340,194' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='194' width='600' height='58' rx='8' fill='#dcfce7' stroke='#334155' stroke-width='1.2'/><rect x='40' y='194' width='14' height='58' rx='4' fill='#16a34a'/><text x='68' y='218' text-anchor='start' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>3. Mạng FRCRN (target noise) và hàm loss</text><text x='68' y='238' text-anchor='start' font-size='12' font-weight='normal' fill='#334155' font-family='Segoe UI,Arial,sans-serif'>Đã rà mã: mask loss trừ SI-SNR trên nhiễu; chưa chạy thử trên molab</text><path d='M340,252 L340,266' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='266' width='600' height='58' rx='8' fill='#fee2e2' stroke='#334155' stroke-width='1.2'/><rect x='40' y='266' width='14' height='58' rx='4' fill='#dc2626'/><text x='68' y='290' text-anchor='start' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>4. Môi trường molab</text><text x='68' y='310' text-anchor='start' font-size='12' font-weight='normal' fill='#334155' font-family='Segoe UI,Arial,sans-serif'>Thiếu soundfile, torchaudio, timm; chưa có dữ liệu và checkpoint pretrain</text><path d='M340,324 L340,338' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='338' width='600' height='58' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.2'/><rect x='40' y='338' width='14' height='58' rx='4' fill='#ca8a04'/><text x='68' y='362' text-anchor='start' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>5. Huấn luyện và validation</text><text x='68' y='382' text-anchor='start' font-size='12' font-weight='normal' fill='#334155' font-family='Segoe UI,Arial,sans-serif'>Train chỉ dùng 2 giây đầu của mỗi clip 4 giây; mới có 1 seed</text><path d='M340,396 L340,410' fill='none' stroke='#334155' stroke-width='1.4' marker-end='url(#a)'/><rect x='40' y='410' width='600' height='58' rx='8' fill='#fef9c3' stroke='#334155' stroke-width='1.2'/><rect x='40' y='410' width='14' height='58' rx='4' fill='#ca8a04'/><text x='68' y='434' text-anchor='start' font-size='14' font-weight='bold' fill='#0f172a' font-family='Segoe UI,Arial,sans-serif'>6. Lưu kết quả</text><text x='68' y='454' text-anchor='start' font-size='12' font-weight='normal' fill='#334155' font-family='Segoe UI,Arial,sans-serif'>Sandbox molab là tạm thời: cần sao lưu Drive và tải kết quả về máy</text><rect x='70' y='488' width='16' height='16' rx='3' fill='#dcfce7' stroke='#16a34a' stroke-width='1.4'/><text x='94' y='501' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>ổn</text><rect x='250' y='488' width='16' height='16' rx='3' fill='#fef9c3' stroke='#ca8a04' stroke-width='1.4'/><text x='274' y='501' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>lưu ý</text><rect x='440' y='488' width='16' height='16' rx='3' fill='#fee2e2' stroke='#dc2626' stroke-width='1.4'/><text x='464' y='501' text-anchor='start' font-size='12' font-weight='normal' fill='#475569' font-family='Segoe UI,Arial,sans-serif'>đang chặn</text></svg>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Checklist trước khi chạy trên molab

    Sandbox molab là môi trường tạm: hết phiên (giới hạn 12 giờ) hoặc bị xoá thì mất toàn bộ `output/`, trừ những gì đã đẩy ra ngoài. **[đọc, theo doc và log của repo]**

    1. Đưa **mã local** lên sandbox: mã `src/` trên molab từng là bản cũ (không có `selection_metric` và `resume_path`) và thiếu `scripts/drive_sync.py`. **[đọc]**
    2. Bật `USE_DRIVE = True` và dùng thư mục Drive **mới** (`check_drive_dir` từ chối các thư mục đã có), để có `resume.pt` và `best.pt`.
    3. Đặt token (Kaggle, W&B) qua biến môi trường, **không** ghi vào ô code.
    4. Tải `result.json`, lịch sử epoch và `best.pt` về máy ngay sau khi chạy xong.
    5. Ghi lại `run_id`, commit (`git rev-parse HEAD`), seed và cấu hình dùng (`SETTINGS`, `BASE_CHOICE`) cùng với kết quả.
    6. Đo 50 bước huấn luyện thật trước khi ước lượng thời gian toàn bộ.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Đã và chưa kiểm chứng

    **Đã kiểm chứng từ mã và tệp trong repo [đọc]**
    - Mạng, mask `tanh + tanh`, loss `mask loss − SI-SNR`, `training_target = noise / scale` khi `target = noise`, `forward` trả thẳng nhiễu.
    - `finetune_frcrn` dùng `Mixture36Dataset(load_noise=True)` nên đúng với "mixture vào, nhiễu làm tham chiếu".
    - Chỉ số chọn best (`noise_si_snr_hi`) và cách tách theo SNR.

    **Chưa kiểm chứng**
    - Hiệu quả của `target = noise` so với `speech` trên dữ liệu này: chưa có lần chạy đủ epoch, đủ seed nào.
    - Thời gian mỗi epoch trên toàn bộ dữ liệu (chỉ có ước tính thô).
    - Công thức nhân SNR của `noise_path` (theo bạn đã áp dụng khi tạo mix-dataset) và việc `mixture = clean + noise` đúng bằng số: không có trong repo. Với `target = noise` điều này quyết định tham chiếu có đúng thang với mixture hay không, nên nên kiểm bằng `RUN_DATA_CHECK` (SNR đo so với nhãn, phần dư `mixture − clean − noise`) trước khi chạy thật.
    - Đẳng thức mask nhiễu + mask tiếng nói = 1 bằng số: kiểm bằng `RUN_CHECKS` (mới được chứng minh đại số).
    - Nguồn gốc của `best.pt` cũ (+1,95 dB ở 15–20 dB): chưa xác nhận do cấu hình nào tạo ra.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Chạy toàn bộ pipeline

    Ô cuối: `RUN_TRAINING = False` mặc định. Đặt `True` để gọi `run_all()`: tải dữ liệu (cần biến môi trường `KAGGLE_ACCESS_TOKEN` nếu chưa có dữ liệu),
    fine-tune `target: noise` trên **tiếng nói (`clean_path`) của mix-dataset trộn động với nhiễu AudioSet 36L** (AudioSet 40L bỏ 4 clip-nhãn Bass drum, Snare drum, Cello, Double bass; SNR rút từ −5…20 dB, validation cố định theo seed; `scripts/dynamic_mix_36l.py`), **hiện tiến trình từng epoch** (thanh tqdm, `train_loss`, `val_loss`,
    `noise_si_snr`, `noise_si_snr_hi`, `lr`, thời gian) và sao lưu `best.pt`/`resume.pt` lên Drive. Chạy trong chính kernel nên kernel bận tới khi xong. Mỗi lần có kết quả tốt nhất mới, ngoài `best.pt` (bản mới nhất) còn lưu `best_e<epoch>_<UTC>.pt` lên Drive; hồ sơ `run_manifest.json` (mốc thời gian từng epoch, vị trí Drive, mã băm code đã dùng) được ghi local và lên Drive. Muốn tái triển khai khi sandbox mất kết nối: `python -m scripts.molab_deploy record|push|launch|status` (xem docstring của `scripts/molab_deploy.py`).
    """)
    return

@app.cell
def _(mo):
    RUN_TRAINING = False  # True: chạy toàn bộ pipeline (tải dữ liệu, fine-tune target noise, speech mix-dataset + nhiễu AudioSet 36L, lưu Drive)
    TRAIN_SETTINGS = {
        "epochs": 40,  # số epoch tối đa; kết thúc sớm nếu validation không cải thiện (patience trong config/frcrn.yaml)
        "max_samples": None,  # số mẫu mỗi epoch; None: toàn bộ (số giọng nói duy nhất của split)
        "use_drive": True,  # sao lưu best.pt, resume.pt, result.json lên Drive (cần token Drive đã đăng nhập)
        "drive_dir": "FA26_Capstone/Truong/frcrn-noise-dyn36-run1",  # thư mục Drive MỚI (chưa tồn tại)
        "work_dir": "output/frcrn-noise-dyn36-run1",  # tương đối so với thư mục gốc repo (/marimo trên sandbox)
        "smoke": False,  # True: 1 epoch, 64 mẫu để kiểm đường chạy
        "workers": 8,
    }

    # Module repo cần nạp lại theo thứ tự phụ thuộc: kernel giữ module đã import nên mã mới đẩy lên đĩa sẽ không tự có hiệu lực
    REPO_MODULES = (
        "scripts.drive_sync",
        "scripts.load_dataset",
        "scripts.download_dataset",
        "scripts.audioset_noise_ref",
        "scripts.dynamic_mix_36l",
        "src.train.frcrn_trainer",
        "src.train",
        "scripts.finetune",
        "scripts.sweep_frcrn",
        "scripts.run_manifest",
        "scripts.run_frcrn_noise",
    )


    def run_all(settings=TRAIN_SETTINGS):
        """Tải dữ liệu, fine-tune FRCRN (target noise; tiếng nói mix-dataset trộn động với nhiễu AudioSet 36L), hiện tiến trình từng epoch, lưu Drive."""
        import importlib
        import os
        import sys
        import time
        from pathlib import Path

        from tqdm.auto import tqdm

        importlib.invalidate_caches()
        reloaded = [name for name in REPO_MODULES if name in sys.modules]
        for name in reloaded:
            importlib.reload(sys.modules[name])
        print(f"đã nạp lại {len(reloaded)} module repo")
        for env_name, marker in (("MLNC_MIXTURE_36L", "labels.txt"), ("MLNC_AUDIOSET_40L", "metadata/manifest.csv")):
            root = os.environ.get(env_name)
            if root and not (Path(root) / marker).is_file():  # gốc sai (Kaggle giải nén lồng một cấp): để ensure_dataset dò lại
                print(f"bỏ {env_name}={root}: thiếu {marker}")
                os.environ.pop(env_name)
        from scripts.run_frcrn_noise import ROOT, main

        work_dir = Path(settings["work_dir"])
        work_dir = work_dir if work_dir.is_absolute() else ROOT / work_dir
        finished = work_dir / "checkpoint" / "frcrn" / "result.json"
        if finished.is_file() and not settings.get("allow_finished", False):
            raise FileExistsError(f"{finished} đã có: lần chạy coi như xong và sẽ không huấn luyện. Đổi work_dir và drive_dir sang thư mục mới.")

        start = time.perf_counter()
        total = settings["epochs"]
        bar = tqdm(total=total, desc="epoch", unit="epoch")

        def on_epoch(epoch, scores):
            bar.update(1)
            bar.set_postfix(
                train_loss=f"{scores['train_loss']:.1f}", val_loss=f"{scores['loss']:.1f}", si_snr_hi=f"{scores['noise_si_snr_hi']:.2f}"
            )
            tqdm.write(
                f"epoch {epoch}/{total} | train_loss {scores['train_loss']:.2f} | val_loss {scores['loss']:.2f}"
                f" | noise_si_snr {scores['noise_si_snr']:.2f} dB | noise_si_snr_hi {scores['noise_si_snr_hi']:.2f} dB"
                f" | lr {scores['lr']:.1e} | {(time.perf_counter() - start) / 60:.1f} phút"
            )

        argv = ["--workers", str(settings["workers"]), "--noise-source", "dynamic36", "--drive-dir", settings["drive_dir"]]
        argv += ["--work-dir", str(work_dir)]
        if settings["use_drive"]:
            argv.append("--use-drive")
        if settings["smoke"]:
            argv.append("--smoke")
        if settings["epochs"]:
            argv += ["--epochs", str(settings["epochs"])]
        if settings["max_samples"]:
            argv += ["--max-samples", str(settings["max_samples"])]
        try:
            return main(argv, on_epoch=on_epoch)
        finally:
            bar.close()


    _result = run_all() if RUN_TRAINING else None
    mo.md(
        "`RUN_TRAINING = False`: chưa chạy. Đặt `True` rồi chạy lại ô này để bắt đầu (xem mục 9 và 12)."
        if _result is None
        else f"Xong: best_epoch **{_result['best_epoch']}**, best_score **{_result['best_score']:.3f}**."
    )
    return REPO_MODULES, RUN_TRAINING, TRAIN_SETTINGS, run_all


if __name__ == "__main__":
    app.run()
