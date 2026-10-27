"""Hiệu chỉnh bias theo lớp (post-hoc logit adjustment) cho student: z'_c = z_c + b_c.

Fit trên val, khoá tham số, rồi áp dụng đúng một lần lên test. Không đổi trọng số.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TAUS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
OVER_RATIO, UNDER_RATIO = 1.25, 0.80
MAX_CLASS_DROP, MIN_MACRO_GAIN = 0.05, 0.02


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion(labels: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    return cm


def metrics(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray | None = None) -> dict:
    """Chỉ số tổng thể, theo lớp và thiên lệch dự đoán."""
    num_classes = logits.shape[1]
    z = logits if bias is None else logits + bias
    pred = z.argmax(1)
    cm = confusion(labels, pred, num_classes)
    true_n, pred_n, hit = cm.sum(1), cm.sum(0), np.diag(cm)
    recall = hit / np.maximum(true_n, 1)
    precision = hit / np.maximum(pred_n, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    ratio = pred_n / np.maximum(true_n, 1)
    top2 = np.argsort(-z, axis=1)[:, :2]
    return {
        "accuracy": float((pred == labels).mean()),
        "top2_accuracy": float((top2 == labels[:, None]).any(1).mean()),
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()),
        "pred_deviation": float(np.abs(ratio - 1).mean()),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "pred_ratio": ratio,
        "confusion": cm,
    }


def prior_bias(pred_prior: np.ndarray, tau: float, mode: str = "A", num_samples: int | None = None) -> np.ndarray:
    """b_c = tau * log(q_c / p_c), q = 1/C. Chế độ A chỉ chỉnh lớp có p/q > 1,25 hoặc < 0,80; chế độ B chỉnh mọi lớp."""
    num_classes = len(pred_prior)
    q = 1.0 / num_classes
    floor = 1.0 / (num_samples or 10 * num_classes)  # lớp chưa từng được dự đoán: tránh log(0)
    bias = tau * np.log(q / np.maximum(pred_prior, floor))
    if mode == "A":
        ratio = pred_prior / q
        bias = np.where((ratio > OVER_RATIO) | (ratio < UNDER_RATIO), bias, 0.0)
    elif mode != "B":
        raise ValueError(f"mode phải là 'A' hoặc 'B', nhận {mode!r}")
    return bias


def violates_constraint(base: dict, new: dict) -> bool:
    """Có lớp nào mất recall quá 0,05 mà macro recall không tăng đủ 0,02 không."""
    worst_drop = float((base["recall"] - new["recall"]).max())
    gain = new["macro_recall"] - base["macro_recall"]
    return worst_drop > MAX_CLASS_DROP and gain < MIN_MACRO_GAIN


def sweep(logits: np.ndarray, labels: np.ndarray, taus=TAUS, modes=("A", "B")) -> tuple[list[dict], dict]:
    """Quét (chế độ, tau) trên val, trả bảng kết quả và baseline."""
    base = metrics(logits, labels)
    prior = base["pred_ratio"] * (np.bincount(labels, minlength=logits.shape[1]) / len(labels))
    rows = []
    for mode in modes:
        for tau in taus:
            bias = prior_bias(prior, tau, mode, len(labels))
            new = metrics(logits, labels, bias)
            rows.append({"mode": mode, "tau": tau, "bias": bias, "metrics": new, "valid": not violates_constraint(base, new)})
    return rows, base


def select(rows: list[dict]) -> dict:
    """Macro recall, rồi macro F1, rồi độ lệch dự đoán nhỏ hơn; bỏ cấu hình vi phạm ràng buộc."""
    candidates = [r for r in rows if r["valid"]] or rows
    return max(candidates, key=lambda r: (round(r["metrics"]["macro_recall"], 6), round(r["metrics"]["macro_f1"], 6), -r["metrics"]["pred_deviation"]))


def accepted(base: dict, chosen: dict, max_bad_classes: int = 2) -> bool:
    """Tiêu chí chấp nhận trên val (mục 8 đặc tả)."""
    new = chosen["metrics"]
    gain_ok = new["macro_recall"] - base["macro_recall"] >= MIN_MACRO_GAIN or new["macro_f1"] - base["macro_f1"] >= 0.015
    bad = int(((base["recall"] - new["recall"]) > MAX_CLASS_DROP).sum())
    return gain_ok and bad <= max_bad_classes and new["pred_deviation"] <= base["pred_deviation"]


def fit_regularized(logits: np.ndarray, labels: np.ndarray, l2: float = 0.1, steps: int = 200) -> np.ndarray:
    """Biến thể mục 5: min CE(z + b) + l2 * ||b||^2, b khởi tạo 0, L-BFGS tối đa `steps` bước."""
    import torch

    z = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    b = torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z + b, y) + l2 * (b**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return b.detach().numpy()


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


@dataclass
class CalibrationConfig:
    mode: str
    tau: float
    class_names: list[str]
    l2: float | None = None
    over_ratio: float = OVER_RATIO
    under_ratio: float = UNDER_RATIO


def save_artifacts(out_dir: Path, config: CalibrationConfig, bias: np.ndarray, before: dict, after: dict, ratio_names: list[str]) -> None:
    """Ghi config, bias, metric val trước/sau, confusion matrix và tỉ lệ dự đoán."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "class_bias.json").write_text(json.dumps(dict(zip(ratio_names, bias.tolist())), ensure_ascii=False, indent=2), encoding="utf-8")
    for tag, m in (("before", before), ("after", after)):
        (out_dir / f"val_metrics_{tag}.json").write_text(json.dumps(_jsonable({k: v for k, v in m.items() if k != "confusion"}), ensure_ascii=False, indent=2), encoding="utf-8")
        np.savetxt(out_dir / f"confusion_matrix_val_{tag}.csv", m["confusion"], fmt="%d", delimiter=",")
    lines = ["class,ratio_before,ratio_after"] + [f"{n},{b:.4f},{a:.4f}" for n, b, a in zip(ratio_names, before["pred_ratio"], after["pred_ratio"])]
    (out_dir / "predicted_count_ratio.csv").write_text("\n".join(lines), encoding="utf-8")


# ---- Phần báo cáo: theo nhóm (SNR, nguồn nhiễu), cặp nhầm ưu tiên, manifest ----

CONFUSABLE_PAIRS = (
    ("Violin", "Clarinet"), ("Chime", "Glockenspiel"), ("Saxophone", "Harmonica"), ("Saxophone", "Trumpet"),
    ("Clarinet", "Trumpet"), ("Acoustic guitar", "Electric piano"), ("Fireworks", "Gunshot"),
    ("Keys jangling", "Writing"), ("Writing", "Computer keyboard"), ("Microwave oven", "Gong"),
)


def group_macro_recall(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray, bias: np.ndarray | None = None) -> dict:
    """Macro recall (trung bình đều trên các lớp có mặt) và số mẫu của từng giá trị trong `groups` (SNR hoặc noise_source_id)."""
    num_classes = logits.shape[1]
    pred = (logits if bias is None else logits + bias).argmax(1)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        true_n = np.bincount(labels[mask], minlength=num_classes)
        hit = np.bincount(labels[mask][pred[mask] == labels[mask]], minlength=num_classes)
        present = true_n > 0
        out[str(g)] = {"n": int(mask.sum()), "macro_recall": float((hit[present] / true_n[present]).mean())}
    return out


def compare_groups(logits, labels, groups, bias, drop_warn: float = 0.02) -> dict:
    """Trước/sau theo nhóm; `worse` liệt kê nhóm bị giảm macro recall quá `drop_warn` (hiệu chỉnh không ổn định)."""
    before, after = group_macro_recall(logits, labels, groups), group_macro_recall(logits, labels, groups, bias)
    rows = {g: {"n": before[g]["n"], "before": before[g]["macro_recall"], "after": after[g]["macro_recall"],
                "delta": after[g]["macro_recall"] - before[g]["macro_recall"]} for g in before}
    return {"groups": rows, "worse": [g for g, r in rows.items() if r["delta"] < -drop_warn]}


def _resolve(name: str, class_names: list[str]) -> int | None:
    key = name.lower()
    exact = [i for i, c in enumerate(class_names) if c.lower() == key]
    prefix = [i for i, c in enumerate(class_names) if c.lower().startswith(key)]
    hit = exact or prefix
    return hit[0] if hit else None


def pair_confusions(cm: np.ndarray, class_names: list[str], pairs=CONFUSABLE_PAIRS) -> list[dict]:
    """Số lần nhầm hai chiều của từng cặp và tỉ lệ trên tổng mẫu thật của hai lớp. Cặp không khớp tên lớp được đánh dấu `resolved: False`."""
    rows = []
    for a, b in pairs:
        ia, ib = _resolve(a, class_names), _resolve(b, class_names)
        if ia is None or ib is None:
            rows.append({"pair": f"{a}-{b}", "resolved": False})
            continue
        ab, ba = int(cm[ia, ib]), int(cm[ib, ia])
        rows.append({"pair": f"{a}-{b}", "resolved": True, "a_as_b": ab, "b_as_a": ba, "total": ab + ba,
                     "rate": (ab + ba) / max(int(cm[ia].sum() + cm[ib].sum()), 1)})
    return rows


def flipped_classes(ratio_before: np.ndarray, ratio_after: np.ndarray) -> list[int]:
    """Lớp 'đảo chiều': từ dự đoán thiếu (< 0,80) sang thừa (> 1,25) hoặc ngược lại."""
    flip = ((ratio_before < UNDER_RATIO) & (ratio_after > OVER_RATIO)) | ((ratio_before > OVER_RATIO) & (ratio_after < UNDER_RATIO))
    return np.nonzero(flip)[0].tolist()


def file_sha256(path: Path, block: int = 1 << 24) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out_dir: Path, checkpoints: dict[str, Path], feature_config: dict, seed: int | None, repo_root: Path | None = None) -> dict:
    """run_manifest.json: git commit, hash checkpoint, cấu hình đặc trưng, ngày chạy, seed. Không có git thì commit = None."""
    import datetime
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {"git_commit": commit, "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "seed": seed,
                "feature_config": feature_config, "checkpoints": {k: {"path": str(v), "sha256": file_sha256(Path(v))} for k, v in checkpoints.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def save_report(out_dir: Path, name: str, report) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
