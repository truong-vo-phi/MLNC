import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell(hide_code=True)
def import_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def diagram_overview(mo):
    mo.vstack([
        mo.md(r"""
    # KD-MoE trên Mixture-36l: từ fine-tune teacher đến student

    Mục tiêu: nhận diện **36 lớp nhiễu** trong tín hiệu nhiễu + tiếng nói ở 6 mức SNR (−5…20 dB) bằng một student nhỏ (KD-MoE, 1–14M tham số) học từ một teacher lớn (BEATs 90M).

    | Giai đoạn | Việc | Đầu ra |
    | --- | --- | --- |
    | 1. Dữ liệu | tải Mixture-36l, AudioSet-40l, VLSP-100h từ Kaggle | thư mục dataset |
    | 2. Pretrain | FRCRN (Drive) + BEATs thuần (HuggingFace) | hai model đóng băng |
    | 3. Fine-tune FRCRN | học tách nhiễu trên Mixture-36l | `frcrn/best.pt` |
    | 4. Fine-tune BEATs | phân loại 36 lớp trên đầu ra FRCRN | `beats/best.pt` |
    | 5. Teacher p_t | embedding BEATs + linear probe cross-fit | `teacher_{split}.npz` |
    | 6. Student KD-MoE | chưng cất p_t, 2 stage | `stage{1,2}/best.pt`, 9 metrics |

    Mọi giai đoạn có **resume + sao lưu Drive**, được đánh giá bằng **9 metrics**: top1, top3, balanced accuracy, precision macro, recall macro, macro F1, micro F1, mAP, macro AUC.
    """),
        mo.mermaid("""
    flowchart LR
        D["Dữ liệu<br/>Mixture-36l / AudioSet-40l / VLSP"] --> P["Pretrain<br/>FRCRN + BEATs"]
        P --> F1["Fine-tune FRCRN<br/>tách nhiễu"]
        F1 --> F2["Fine-tune BEATs<br/>mask bằng FRCRN"]
        F2 --> T["Teacher p_t<br/>embedding + probe cross-fit"]
        T --> S["Student KD-MoE<br/>stage 1 → stage 2"]
        S --> E["Đánh giá<br/>9 metrics"]
    """),
    ])

    return


@app.cell
def gg_drive(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    mo.output.clear_console()
    return (fs,)


@app.cell(hide_code=True)
def diagram_data(mo):
    mo.vstack([
        mo.md(r"""
    ## 1. Dữ liệu

    - **Mixture-36l**: mixture = tiếng nói + nhiễu, nhãn đơn lớp (36 lớp nhiễu), 6 mức SNR × 5040 mẫu train; chia sẵn `train / validation / test` (30 240 / 6 480 / 6 480). Mỗi clip 4 s, 16 kHz. Có kèm `noise` (nhiễu thuần) để huấn luyện FRCRN.
    - **AudioSet-40l**: nhiễu thuần 40 lớp (multi-hot), clip 10 s. Dùng để tính **RMS chuẩn r** (trung vị RMS nhiễu thuần) cho bước teacher.
    - **VLSP-100h**: tiếng nói thuần, split cố định theo hash đường dẫn.
    - Tải bằng token Kaggle, có resume `.part`; đã giải nén thì bỏ qua. Root dataset đọc từ biến môi trường `MLNC_*` hoặc `config/url.yaml`.
    """),
        mo.mermaid("""
    flowchart LR
        K["Kaggle API<br/>zip, resume .part"] --> Z["Giải nén"]
        Z --> M["Mixture-36l<br/>mixture + noise + nhãn 36"]
        Z --> A["AudioSet-40l<br/>nhiễu thuần 40 lớp"]
        Z --> V["VLSP-100h<br/>tiếng nói thuần"]
        M --> TR["train 30 240 / val 6 480 / test 6 480"]
        A --> R["RMS chuẩn r"]
    """),
    ])

    return


@app.cell
def download_data_from_kaggle():
    # Tải bộ dataset từ Kaggle bằng src/utils/download.py
    # (dataset nào đã tải/giải nén rồi thì hàm tự bỏ qua, chạy lại an toàn)
    from src.utils.download import (
        download_and_extract_audioset,
        download_and_extract_mix_dataset,
        download_and_extract_vlsp,
    )

    is_vlsp_downloaded = True
    is_audioset_downloaded = True
    is_mixer_downloaded = True

    vlsp_dir = audioset_dir = mix_dataset_dir = None

    if is_vlsp_downloaded is True:
        vlsp_dir = download_and_extract_vlsp()
    if is_audioset_downloaded is True:
        audioset_dir = download_and_extract_audioset()
    if is_mixer_downloaded is True:
        mix_dataset_dir = download_and_extract_mix_dataset()
    return audioset_dir, mix_dataset_dir, vlsp_dir


@app.cell(hide_code=True)
def diagram_pretrain(mo):
    mo.vstack([
        mo.md(r"""
    ## 2. Nạp pretrain

    - **FRCRN** (`FA26_Capstone/Truong/FRCRN/best.pt` trên Drive → `pretrained/frcrn_beat_v1_best.pt`): mạng tách nhiễu miền phức, đã train trước.
    - **BEATs** (`BEATs_iter3_plus_AS2M.pt`, HuggingFace): bản **pretrain thuần**, chưa fine-tune trên nhãn AudioSet nên không rò nhãn; đầu 36 lớp khởi tạo ngẫu nhiên.
    - Cả hai nạp bằng `load_pretrained` (báo missing/unexpected key) rồi đóng băng. `device` tự chọn CUDA nếu có.
    """),
        mo.mermaid("""
    flowchart LR
        G["Google Drive<br/>FRCRN/best.pt"] --> FR["FRCRNSeparator<br/>load_pretrained"]
        H["HuggingFace<br/>BEATs_iter3_plus_AS2M"] --> BE["BEATClassifier<br/>load_pretrained"]
        FR --> Z["freeze, eval"]
        BE --> Z
    """),
    ])

    return


@app.cell
def load_pretrain(fs):
    # Tải và nạp pretrain: FRCRN (FA26_Capstone/Truong/FRCRN/best.pt trên Drive) + BEATs (pretrain, từ HuggingFace)
    import torch

    from scripts.load_pretrain import load_teacher_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separator, beat_classifier = load_teacher_models(fs, device)

    f"device={device} | FRCRN (target={separator.config.target}) {separator.num_parameters() / 1e6:.1f}M | BEATs {beat_classifier.num_parameters() / 1e6:.1f}M (đã đóng băng)"

    return beat_classifier, device, separator


@app.cell
def show_config(
    audioset_dir,
    beat_classifier,
    mix_dataset_dir,
    mo,
    separator,
    vlsp_dir,
):
    # Hiển thị toàn bộ config: model (tham số, cấu trúc), pipeline và dataset
    from scripts.describe_config import describe_pipeline

    config_report = describe_pipeline(
        separator,
        beat_classifier,
        dataset_dirs={"mixture-36l": mix_dataset_dir, "audioset-40l": audioset_dir, "vlsp-100h": vlsp_dir},
    )
    mo.accordion({title: mo.md(body) for title, body in config_report.items()})

    return


@app.cell(hide_code=True)
def diagram_frcrn_ft(mo):
    mo.vstack([
        mo.md(r"""
    ## 3. Fine-tune FRCRN trên Mixture-36l

    **Kiến trúc FRCRN**: STFT (cửa sổ 640, bước 320, 321 bin) → **2 UNet xếp tầng** (128 kênh, FSMN bậc 20) → mặt nạ phức **cIRM** `tanh(u1) + tanh(u2(u1))` → iSTFT → ước lượng nhiễu $\hat n$.

    **Huấn luyện**: $\hat n = \mathrm{FRCRN}(x)$, đích là nhiễu thật $n$ đi kèm mixture; mất mát SI-SNR. Đoạn train 2 s, đánh giá 4 s, Adam lr $10^{-4}$, grad-clip 10, **giảm nửa lr** khi 3 epoch không cải thiện (nạp lại best), tối đa 40 epoch, patience 15. Tiêu chí chọn: `noise_si_snr` trên validation. Có resume mỗi epoch + `best.pt`/`result.json` sao lưu Drive.
    """),
        mo.mermaid("""
    flowchart LR
        X["mixture x<br/>2 s"] --> ST["STFT"]
        ST --> U1["UNet 1<br/>FSMN"]
        U1 --> U2["UNet 2<br/>FSMN"]
        U2 --> MK["mask cIRM"]
        MK --> IS["iSTFT"]
        IS --> NH["ước lượng nhiễu n_hat"]
        N["nhiễu thật n"] --> L["loss SI-SNR"]
        NH --> L
        L --> OPT["Adam, lr 1e-4<br/>halve lr khi stale"]
    """),
    ])

    return


@app.cell(hide_code=True)
def diagram_beats_ft(mo):
    mo.vstack([
        mo.md(r"""
    ## 4. Fine-tune BEATs với mask FRCRN (kịch bản `all + mask`)

    **Luồng**: $x \rightarrow$ FRCRN (đóng băng) $\rightarrow \hat n \rightarrow$ Fbank 128 mel → **BEATs** (patch 16×16, 12 lớp, 12 head, dim 768, 192 patch/clip 4 s) → mean-pool → Linear → 36 logit.

    **Huấn luyện**: fine-tune **toàn bộ** BEATs (90M tham số), cross-entropy, AdamW lr $5\cdot10^{-5}$, weight-decay 0.01, betas (0.9, 0.98), cosine + warmup 10%, batch 32, tối đa 100 epoch, patience 15, chọn theo `macro_f1`. Mỗi epoch validation đủ 9 metrics; cuối cùng đánh giá test + ma trận nhầm lẫn.
    """),
        mo.mermaid("""
    flowchart LR
        X["mixture x<br/>4 s"] --> FR["FRCRN<br/>đóng băng"]
        FR --> NH["n_hat"]
        NH --> FB["Fbank<br/>128 mel × 398"]
        FB --> PE["PatchEmbed 16x16<br/>192 patch"]
        PE --> EN["Encoder 12 lớp<br/>dim 768"]
        EN --> MP["mean-pool"]
        MP --> HD["Linear 36"]
        HD --> CE["cross-entropy"]
    """),
    ])

    return


@app.cell
def finetune(beat_classifier, device, fs, mo, separator):
    # Fine-tune FRCRN rồi BEATs (mask bằng FRCRN) trên Mixture-36l, mọi mức SNR
    # Epoch/patience lấy từ config/frcrn.yaml (40/15) và config/beat.yaml (100/15).
    # Resume + best.pt + result.json tự đồng bộ lên Drive (FA26_Capstone/Truong/kd-moe-ft/{frcrn,beats}).
    import logging
    from pathlib import Path

    from scripts.finetune import METRIC_NAMES, finetune_beats, finetune_frcrn, prepare_stage

    logger = logging.getLogger("kd-moe")
    FT_DIR = Path("pretrained") / "finetune"

    frcrn_files = prepare_stage("frcrn", FT_DIR, logger, fs)
    beats_files = prepare_stage("beats", FT_DIR, logger, fs)

    frcrn_result = finetune_frcrn(separator, device, logger, frcrn_files)
    beats_result = finetune_beats(beat_classifier, separator, device, logger, beats_files)

    mo.ui.table([{"metric": name, "test": round(beats_result["test"][name], 4)} for name in METRIC_NAMES])

    return METRIC_NAMES, logger


@app.cell
def load_finetuned(beat_classifier, device, fs, logger, separator):
    # Nạp FRCRN + BEATs đã fine-tune (best.pt; tự khôi phục từ Drive nếu thiếu) và đóng băng
    from scripts.run_kdmoe import build_teacher_cache, load_finetuned, train_student

    finetuned_separator, finetuned_beats = load_finetuned(separator, beat_classifier, device, logger, fs=fs)

    return (
        build_teacher_cache,
        finetuned_beats,
        finetuned_separator,
        train_student,
    )


@app.cell(hide_code=True)
def diagram_teacher(mo):
    mo.vstack([
        mo.md(r"""
    ## 5. Teacher p_t (không rò nhãn)

    1. **T1** — đo SI-SDR theo SNR để chọn $\hat n$ là đầu ra FRCRN hay $x-\text{đầu ra}$.
    2. **T3** — $\hat n$ được chuẩn hoá RMS về $r$ (trung vị RMS nhiễu AudioSet), đưa qua BEATs đã fine-tune, lấy embedding **mean-pool 768 chiều** (fp16).
    3. **T4/T5** — **linear probe cross-fit 5 fold** (nhóm theo `noise_ytid` để không lẫn clip nhiễu giữa fold): train dùng logit **out-of-fold**; validation/test dùng trung bình 5 probe. Lưu `teacher_{split}.npz` (logit 36, fp16) + `teacher_report.json` (9 metrics OOF, router acc theo SNR).

    Lý do dùng probe cross-fit: BEATs đã thấy toàn bộ tập train, lấy logit trực tiếp sẽ tự tin giả; logit out-of-fold giữ p_t trung thực.
    """),
        mo.mermaid("""
    flowchart LR
        X["mixture x"] --> FR["FRCRN"]
        FR --> RN["chuẩn hoá RMS về r"]
        RN --> BE["BEATs fine-tuned<br/>mean-pool 768"]
        BE --> PR["linear probe<br/>5 fold, nhóm theo noise_ytid"]
        PR --> OOF["train: logit out-of-fold"]
        PR --> ENS["val / test: trung bình 5 probe"]
        OOF --> C["teacher_split.npz<br/>logit 36 fp16"]
        ENS --> C
    """),
    ])

    return


@app.cell
def teacher_cache(
    METRIC_NAMES,
    build_teacher_cache,
    device,
    finetuned_beats,
    finetuned_separator,
    fs,
    logger,
    mo,
):
    # Teacher p_t: FRCRN -> BEATs embedding -> cross-fit probe, cache vào pretrained/finetune/teacher_cache (sao lưu Drive)
    teacher_report = build_teacher_cache(finetuned_separator, finetuned_beats, device, logger, fs=fs)
    mo.ui.table([{"metric": name, "oof": round(teacher_report[name], 4)} for name in METRIC_NAMES])

    return (teacher_report,)


@app.cell(hide_code=True)
def diagram_student(mo):
    mo.vstack([
        mo.md(r"""
    ## 6. Student KD-MoE

    **Kiến trúc** (cấu hình S: d=128, 1.28M tham số; M 6.45M; L 14.31M):
    log-mel (128×384) → SpecAugment → patch embed → **khối lai** (attention + GRU) → **cổng mặt nạ** $g$ (+ head phụ dự đoán log-mel nhiễu) → **khối ViT** → LayerNorm → gộp tần số + BiGRU → embedding $e$ → **Router** (4 nhóm) và **4 expert** (mỗi expert ra 36 logit). Đầu ra: $\text{logits}=\sum_k \mathrm{softmax}(r)_k \cdot \text{expert}_k$ (inference `weighted`).

    **Mất mát (stage 2)**:
    $$\mathcal{L} = \mathrm{CE}_{\text{ls}=0.1} + 1.0\,\mathrm{KL}_{T=2}\!\big(\text{router},\ p_t\cdot M\big) + 1.0\,L_{\text{front}} + 0.1\,L_{\text{cons}}$$

    - $\mathrm{KL}$: chưng cất từ p_t của teacher, **gom 36 lớp → 4 nhóm** bằng ma trận thành viên $M$.
    - $L_{\text{front}}$: L1 bất đối xứng trên log-mel nhiễu (phạt ước lượng thừa ×2).
    - $L_{\text{cons}}$: $1-\cos$ giữa embedding của các mix cùng clip nhiễu.

    **Huấn luyện**: stage 1 (8 epoch) chỉ $L_{\text{front}}$ cho patch/khối lai/cổng/head phụ; stage 2 khởi tạo từ best stage 1, toàn bộ loss. AdamW lr $5\cdot10^{-4}$, wd 0.05, bf16, EMA 0.999, tối đa 120 epoch, patience 15, chọn theo `macro_acc` (trung bình balanced accuracy 6 mức SNR). Batch ghép cặp 32 clip nhiễu. Đánh giá cuối: 9 metrics trên test + router acc, gate, mAP theo SNR.
    """),
        mo.mermaid("""
    flowchart TB
        W["mixture x"] --> LM["log-mel 128x384"]
        NZ["nhiễu thật n<br/>khi train"] -.-> LM
        LM --> SA["SpecAugment"]
        SA --> PE["PatchEmbed"]
        PE --> HY["Khối lai x n_hybrid<br/>attention + GRU"]
        HY --> GT["Cổng mặt nạ g"]
        GT --> NHD["head phụ: log-mel nhiễu<br/>L_front"]
        GT --> VT["Khối ViT x n_vit"]
        VT --> LN["LayerNorm"]
        LN --> PG["gộp tần số + BiGRU"]
        PG --> EMB["embedding e<br/>L_cons"]
        EMB --> RT["Router 4 nhóm"]
        EMB --> EX["4 Expert x 36 logit"]
        RT --> MIX["tổng có trọng số softmax"]
        EX --> MIX
        MIX --> OUT["logits 36<br/>CE + KD"]
        PT["p_t teacher<br/>gom 4 nhóm"] -.-> RT
    """),
    ])

    return


@app.cell
def kdmoe_student(
    METRIC_NAMES,
    device,
    fs,
    logger,
    mo,
    teacher_report,
    train_student,
):
    # KD-MoE student: stage 1 (L_front) -> stage 2 (toàn bộ loss); resume + best + result sao lưu Drive
    assert teacher_report, "cần teacher cache trước"
    kdmoe_result = train_student(device, logger, fs=fs)
    mo.ui.table([{"metric": name, "test": round(kdmoe_result["stage2"]["test"][name], 4)} for name in METRIC_NAMES])

    return


if __name__ == "__main__":
    app.run()
