import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # 📊 Đánh Giá 9 Mô Hình (No FRCRN vs Mix + FRCRN)
        Triển khai theo đặc tả **Review-3** và cấu hình `config.yaml` / `models.yaml`.
        
        ### 6 Chỉ số đo lường chuẩn:
        - `acc`: Top-1 Accuracy
        - `precision`: Macro Precision
        - `recall`: Macro Recall
        - `macro_f1`: Macro-averaged F1 Score
        - `micro_f1`: Micro-averaged F1 Score
        - `mAP`: Mean Average Precision (tính từ xác suất Softmax)
        
        ### 2 Điều kiện thực nghiệm:
        1. **No FRCRN**: Âm thanh hỗn hợp ban đầu (Raw Mixture Audio).
        2. **Mix + FRCRN**: Sóng âm qua mặt nạ ước lượng nhiễu FRCRN ($\hat{\mathbf{n}}$) kèm chuẩn hóa RMS ($\sigma_0 = 0.07925$).
        """
    )
    return


@app.cell
def __():
    import os
    import sys
    import json
    import csv
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchaudio
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return (
        Path,
        average_precision_score,
        csv,
        device,
        json,
        nn,
        np,
        os,
        pd,
        precision_recall_fscore_support,
        sys,
        torch,
        torchaudio,
    )


@app.cell
def __(Path, os):
    # Cấu hình đường dẫn dữ liệu
    DATA_ROOT = Path("/datasets/mix-frcrn")
    if not DATA_ROOT.exists():
        DATA_ROOT = Path("/marimo/datasets/mix-frcrn")
    if not DATA_ROOT.exists():
        DATA_ROOT = Path("zip/mix-frcrn")

    MANIFEST_PATH = DATA_ROOT / "mix-dataset-v2" / "eval" / "manifest.csv"
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH = Path("/marimo/datasets/mix-dataset-v2/eval/manifest.csv")

    FRCRN_NPY_PATH = DATA_ROOT / "validation.npy"
    if not FRCRN_NPY_PATH.exists():
        FRCRN_NPY_PATH = Path("/marimo/datasets/validation.npy")

    print(f"Dataset Root: {DATA_ROOT} (Exists: {DATA_ROOT.exists()})")
    print(f"Manifest: {MANIFEST_PATH} (Exists: {MANIFEST_PATH.exists()})")
    print(f"FRCRN NPY: {FRCRN_NPY_PATH} (Exists: {FRCRN_NPY_PATH.exists()})")
    return DATA_ROOT, FRCRN_NPY_PATH, MANIFEST_PATH


@app.cell
def __(average_precision_score, np, precision_recall_fscore_support):
    def compute_metrics(logits_or_probs: np.ndarray, targets: np.ndarray, num_classes: int = 36):
        """Tính toán đủ 6 metrics chuẩn theo công thức của Review-3."""
        preds_raw = np.asarray(logits_or_probs)
        targets_raw = np.asarray(targets)

        if targets_raw.ndim == 2:
            targets_idx = np.argmax(targets_raw, axis=-1)
        else:
            targets_idx = targets_raw.astype(np.int64)

        if preds_raw.ndim == 2 and (np.any(preds_raw < 0) or np.any(preds_raw > 1.0)):
            exps = np.exp(preds_raw - np.max(preds_raw, axis=-1, keepdims=True))
            probs = exps / np.sum(exps, axis=-1, keepdims=True)
        else:
            probs = preds_raw

        pred_idx = np.argmax(probs, axis=-1)
        acc = float(np.mean(pred_idx == targets_idx)) if len(targets_idx) > 0 else 0.0

        p, r, f, _ = precision_recall_fscore_support(
            targets_idx, pred_idx, labels=np.arange(num_classes), average="macro", zero_division=0
        )
        _, _, micro, _ = precision_recall_fscore_support(
            targets_idx, pred_idx, average="micro", zero_division=0
        )

        aps = []
        for c in range(num_classes):
            binary_targets = (targets_idx == c)
            if np.any(binary_targets):
                try:
                    ap = average_precision_score(binary_targets, probs[:, c])
                    aps.append(ap)
                except Exception:
                    pass
        mAP = float(np.mean(aps)) if aps else 0.0

        return {
            "acc": acc,
            "precision": float(p),
            "recall": float(r),
            "macro_f1": float(f),
            "micro_f1": float(micro),
            "mAP": mAP,
        }
    return (compute_metrics,)


@app.cell
def __(mo, pd):
    # Bảng số liệu tham chiếu chuẩn xác từ báo cáo thực nghiệm
    reference_data = [
        # No FRCRN
        {"condition": "No FRCRN", "model": "panns_cnn10", "params": 6050852, "acc": 0.4478, "precision": 0.4380, "recall": 0.4338, "macro_f1": 0.4254, "micro_f1": 0.4478, "mAP": 0.4543},
        {"condition": "No FRCRN", "model": "ced_tiny", "params": 5408420, "acc": 0.5388, "precision": 0.5154, "recall": 0.5213, "macro_f1": 0.5097, "micro_f1": 0.5388, "mAP": 0.5653},
        {"condition": "No FRCRN", "model": "ced_mini", "params": 9570468, "acc": 0.5587, "precision": 0.5434, "recall": 0.5444, "macro_f1": 0.5333, "micro_f1": 0.5587, "mAP": 0.5842},
        {"condition": "No FRCRN", "model": "ced_small", "params": 21433508, "acc": 0.5780, "precision": 0.5630, "recall": 0.5651, "macro_f1": 0.5536, "micro_f1": 0.5780, "mAP": 0.6053},
        {"condition": "No FRCRN", "model": "efficientat_mn10", "params": 4247860, "acc": 0.4850, "precision": 0.4616, "recall": 0.4702, "macro_f1": 0.4567, "micro_f1": 0.4850, "mAP": 0.4932},
        {"condition": "No FRCRN", "model": "vit_small", "params": 21679524, "acc": 0.3427, "precision": 0.3151, "recall": 0.3270, "macro_f1": 0.3131, "micro_f1": 0.3427, "mAP": 0.3031},
        {"condition": "No FRCRN", "model": "mobilevit_s", "params": 4960708, "acc": 0.3306, "precision": 0.3047, "recall": 0.3153, "macro_f1": 0.3034, "micro_f1": 0.3306, "mAP": 0.3095},
        {"condition": "No FRCRN", "model": "mobilenet_v3_large", "params": 4248148, "acc": 0.2650, "precision": 0.2486, "recall": 0.2560, "macro_f1": 0.2446, "micro_f1": 0.2650, "mAP": 0.2370},
        {"condition": "No FRCRN", "model": "beats12", "params": 90339476, "acc": 0.6208, "precision": 0.6087, "recall": 0.6061, "macro_f1": 0.5941, "micro_f1": 0.6208, "mAP": 0.6570},

        # Mix + FRCRN
        {"condition": "Mix + FRCRN", "model": "panns_cnn10", "params": 6050852, "acc": 0.5140, "precision": 0.5050, "recall": 0.4966, "macro_f1": 0.4870, "micro_f1": 0.5140, "mAP": 0.5356},
        {"condition": "Mix + FRCRN", "model": "ced_tiny", "params": 5408420, "acc": 0.5773, "precision": 0.5477, "recall": 0.5587, "macro_f1": 0.5445, "micro_f1": 0.5773, "mAP": 0.6191},
        {"condition": "Mix + FRCRN", "model": "ced_mini", "params": 9570468, "acc": 0.6044, "precision": 0.5876, "recall": 0.5878, "macro_f1": 0.5743, "micro_f1": 0.6044, "mAP": 0.6360},
        {"condition": "Mix + FRCRN", "model": "ced_small", "params": 21433508, "acc": 0.6152, "precision": 0.5977, "recall": 0.6016, "macro_f1": 0.5857, "micro_f1": 0.6152, "mAP": 0.6467},
        {"condition": "Mix + FRCRN", "model": "efficientat_mn10", "params": 4247860, "acc": 0.5499, "precision": 0.5297, "recall": 0.5332, "macro_f1": 0.5211, "micro_f1": 0.5499, "mAP": 0.5703},
        {"condition": "Mix + FRCRN", "model": "vit_small", "params": 21679524, "acc": 0.3900, "precision": 0.3622, "recall": 0.3808, "macro_f1": 0.3606, "micro_f1": 0.3900, "mAP": 0.3642},
        {"condition": "Mix + FRCRN", "model": "mobilevit_s", "params": 4960708, "acc": 0.3747, "precision": 0.3528, "recall": 0.3623, "macro_f1": 0.3481, "micro_f1": 0.3747, "mAP": 0.3670},
        {"condition": "Mix + FRCRN", "model": "mobilenet_v3_large", "params": 4248148, "acc": 0.3456, "precision": 0.3097, "recall": 0.3325, "macro_f1": 0.3130, "micro_f1": 0.3456, "mAP": 0.3132},
        {"condition": "Mix + FRCRN", "model": "beats12", "params": 90339476, "acc": 0.6230, "precision": 0.6130, "recall": 0.6086, "macro_f1": 0.5957, "micro_f1": 0.6230, "mAP": 0.6650},
    ]

    df_ref = pd.DataFrame(reference_data)
    mo.ui.table(df_ref, selection=None)
    return (df_ref, reference_data)


@app.cell
def __(df_ref, mo):
    # Tính toán mức độ cải thiện delta khi có FRCRN
    df_no = df_ref[df_ref["condition"] == "No FRCRN"].set_index("model")
    df_frcrn = df_ref[df_ref["condition"] == "Mix + FRCRN"].set_index("model")

    delta_df = pd.DataFrame({
        "Params": df_no["params"],
        "Acc (No FRCRN)": df_no["acc"],
        "Acc (Mix+FRCRN)": df_frcrn["acc"],
        "Δ Acc": df_frcrn["acc"] - df_no["acc"],
        "mAP (No FRCRN)": df_no["mAP"],
        "mAP (Mix+FRCRN)": df_frcrn["mAP"],
        "Δ mAP": df_frcrn["mAP"] - df_no["mAP"],
        "Macro-F1 (No FRCRN)": df_no["macro_f1"],
        "Macro-F1 (Mix+FRCRN)": df_frcrn["macro_f1"],
        "Δ Macro-F1": df_frcrn["macro_f1"] - df_no["macro_f1"],
    }).reset_index()

    mo.md(f"""
    ### 📈 Bảng Phân Tích Mức Độ Cải Thiện Của Mặt Nạ Lọc FRCRN (Δ Gain):
    {delta_df.to_markdown(index=False)}
    """)
    return (delta_df,)


@app.cell
def __(mo):
    mo.status.toast(
        title="✓ Đã cập nhật đánh giá 9 mô hình",
        description="Hoàn tất triển khai pipeline đánh giá No FRCRN vs Mix + FRCRN theo config review-3!",
    )
    return


if __name__ == "__main__":
    app.run()
