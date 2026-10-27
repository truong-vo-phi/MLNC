"""Student ViT-S (bias vị trí có cổng) chưng cất từ BEATs: CE + KD logit, tuỳ chọn q/k/v và bias.

    python -m scripts.run_vits_kd --noise-cache /marimo/cache/nhat --name r2 --qkv-weight 1 --bias-weight 1

Dừng sớm theo macro_acc validation; `best.pt` chọn theo điểm = macro_acc - phạt * chênh lệch train/val. Không đụng tới test.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import build_dataloader, subsample  # noqa: E402
from scripts.run_reviewkd_deit import CONFIG, build_cached_loaders, load_teacher_checkpoint  # noqa: E402
from src.loss.beat_attention_kd import AttentionKD  # noqa: E402
from src.model.baseline.vit_bias import ViTSBias, load_ced_tiny  # noqa: E402
from src.model.kdmoe.frontend import LogMel, SpecAugment  # noqa: E402
from src.train import ReviewKDTrainConfig  # noqa: E402
from src.train.beat_trainer import cosine_with_warmup  # noqa: E402
from src.train.reviewkd_trainer import snr_scores  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

NUM_CLASSES = 36
REPORT = ("top1_accuracy", "balanced_accuracy", "macro_f1", "mAP", "macro_auc")


class Student(nn.Module):
    """n^ (waveform đã chuẩn hoá RMS) -> log-mel -> SpecAugment (chỉ lúc train) -> ViT-S có bias."""

    def __init__(self, frontend: dict, specaug: dict, **vit_kwargs) -> None:
        super().__init__()
        self.frontend = LogMel(**frontend)
        self.spec_augment = SpecAugment(**specaug)
        self.vit = ViTSBias(**vit_kwargs)

    def forward(self, wave: Tensor) -> Tensor:
        with torch.autocast(wave.device.type, enabled=False):
            x, _ = self.frontend(wave.float())
        return self.vit(self.spec_augment(x))


def print_params(student: Student) -> None:
    groups = student.vit.describe_params()
    total = sum(p.numel() for p in student.parameters())
    assert sum(groups.values()) == total, (sum(groups.values()), total)
    print("PARAMS student ViT-S có bias:")
    for name, count in groups.items():
        print(f"PARAMS  {name:36s} {count:>12,d}  {100 * count / total:5.1f}%")
    print(f"PARAMS  {'TỔNG':36s} {total:>12,d}  ({total / 1e6:.2f}M)", flush=True)


def teacher_attentions(teacher) -> list[nn.Module]:
    return [layer.self_attn for layer in teacher.backbone.encoder.layers]


@torch.no_grad()
def evaluate(student: Student, front, loader, device) -> dict[str, float]:
    student.eval()
    probabilities, labels, snrs = [], [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = student(front(batch["waveform"].to(device)))
        probabilities.append(logits.float().softmax(-1).cpu())
        labels.append(batch["target"].argmax(dim=1))
        snrs.append(batch["snr_db"])
    return snr_scores(torch.cat(probabilities).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), NUM_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-cache", type=Path, default=None)
    parser.add_argument("--name", default="show-params")
    parser.add_argument("--show-params", action="store_true", help="chỉ in bảng tham số của student rồi thoát")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--qkv-weight", type=float, default=0.0)
    parser.add_argument("--qkv-loss", choices=("mse", "kl"), default="mse", help="khoảng cách giữa ma trận quan hệ q/k/v của student và teacher")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("plateau", "cosine"), default="plateau",
                        help="plateau: giảm lr một nửa khi macro_acc validation không tăng sau --lr-patience epoch")
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--bias-weight", type=float, default=0.0, help="MSE hệ số cổng s")
    parser.add_argument("--pos-weight", type=float, default=0.0, help="MSE bias vị trí P (bảng khởi tạo bằng nhóm head của teacher)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--arch", default="vit_small_patch16_224", help="kiến trúc timm của student, ví dụ deit_tiny_patch16_224")
    parser.add_argument("--ced-tiny", action="store_true", help="nạp trọng số CED-Tiny (mispeech/ced-tiny) thay cho ImageNet; dùng với --arch deit_tiny_patch16_224")
    parser.add_argument("--bias-init", choices=("teacher", "zero"), default="teacher",
                        help="khởi tạo bảng bias vị trí: nhóm trung bình từ teacher hoặc bằng 0 (không làm lệch trọng số tiền huấn luyện)")
    parser.add_argument("--n-mels", type=int, default=128, help="số dải mel của student (CED-Tiny dùng 64)")
    parser.add_argument("--post-ln", action="store_true", help="thêm LN(alpha * x + sublayer) kiểu BEATs sau mỗi sublayer, alpha = (2N)^0,25")
    parser.add_argument("--no-pre-ln", action="store_true", help="bỏ LayerNorm 1 và 2 trước attention và FFN (chỉ có tác dụng với --post-ln)")
    parser.add_argument("--gap-penalty", type=float, default=0.2)
    parser.add_argument("--train-eval-clips", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "vits-kd")
    args = parser.parse_args()

    raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = raw["model"]
    specaug = dict(freq_masks=model["specaug_freq_masks"], freq_width=max(1, model["specaug_freq_width"] * args.n_mels // 128),
                   time_masks=model["specaug_time_masks"], time_width=model["specaug_time_width"])
    frontend = {**raw["frontend"], "n_mels": args.n_mels}
    vit_kwargs = dict(num_classes=NUM_CLASSES, pretrained=not (args.no_pretrained or args.ced_tiny), freq_bins=args.n_mels, dropout=args.dropout,
                      attn_dropout=args.attn_dropout, drop_path=args.drop_path,
                      post_ln=args.post_ln, pre_ln=not args.no_pre_ln, arch=args.arch)
    if args.show_params:
        print_params(Student(frontend, specaug, **{**vit_kwargs, "pretrained": False}))
        return
    if args.noise_cache is None:
        parser.error("--noise-cache is required unless --show-params")
    config = ReviewKDTrainConfig.from_dict({**raw["train"], "batch_size": args.batch_size})
    loaders = build_cached_loaders(raw, config, args.noise_cache, args.max_samples, 4)
    front, teacher, _ = load_teacher_checkpoint(ROOT / raw["teacher"]["checkpoint"], device, use_frcrn=False)
    teacher.freeze()
    train_eval = build_dataloader(subsample(loaders["train"].dataset, args.train_eval_clips), 64, False, 4)

    student = Student(frontend, specaug, **vit_kwargs)
    if args.ced_tiny:
        print("CED-Tiny", load_ced_tiny(student.vit), flush=True)
    student = student.to(device)
    teacher_layers = teacher_attentions(teacher)
    if args.bias_init == "teacher":
        student.vit.init_bias_from_teacher(teacher_layers[0].relative_attention_bias.weight.detach().cpu())
    else:
        student.vit.attentions[0].relative_attention_bias.weight.data.zero_()
    student_layers = student.vit.attentions
    if len(student_layers) != len(teacher_layers):
        raise ValueError(f"{len(student_layers)} student blocks vs {len(teacher_layers)} teacher layers")
    kd = AttentionKD(args.qkv_weight, args.bias_weight, args.pos_weight, args.qkv_loss)

    out_dir = args.output_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"RUN {args.name} | student {args.arch} {params:.2f}M params | args {vars(args)}", flush=True)
    print_params(student)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    cosine = cosine_with_warmup(optimizer, args.epochs * len(loaders["train"]), 0.1, 0.0) if args.lr_schedule == "cosine" else None
    base_lr, global_step, lr_stale, lr_best = args.lr, 0, 0, -1.0
    best, best_epoch, stale = -1e9, 0, 0
    best_val, best_val_epoch = -1.0, 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        student.vit.set_record(kd.enabled)
        for attention in teacher_layers:
            attention.record = kd.enabled
        started, sums, seen = time.time(), {"loss": 0.0, "correct": 0.0, "qkv": 0.0, "scale": 0.0, "pos": 0.0}, 0
        for batch in loaders["train"]:
            wave = front(batch["waveform"].to(device))
            target = batch["target"].to(device).float()
            if args.mixup > 0 and torch.rand(()) < args.mixup:
                lam = torch.distributions.Beta(0.4, 0.4).sample((wave.shape[0], 1)).to(device)
                lam = torch.maximum(lam, 1 - lam)
                order = torch.randperm(wave.shape[0], device=device)
                wave = lam * wave + (1 - lam) * wave[order]
                target = lam * target + (1 - lam) * target[order]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = student(wave).float()
                smooth = target * (1 - args.label_smoothing) + args.label_smoothing / NUM_CLASSES
                loss = -(smooth * F.log_softmax(logits, -1)).sum(-1).mean()
                if args.logit_weight > 0 or kd.enabled:
                    with torch.no_grad():
                        teacher_logits = teacher(wave).logits.float()
                if args.logit_weight > 0:
                    loss = loss + args.logit_weight * args.temperature**2 * F.kl_div(
                        F.log_softmax(logits / args.temperature, -1), F.softmax(teacher_logits / args.temperature, -1), reduction="batchmean")
            if kd.enabled:
                terms = kd([a.cache for a in student_layers], [a.cache for a in teacher_layers])
                loss = loss + kd.total(terms)
                for key in ("qkv", "scale", "pos"):
                    sums[key] += terms[key].item() * len(target)
            if cosine is None:
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * min(1.0, (global_step + 1) / args.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if cosine is not None:
                cosine.step()
            global_step += 1
            sums["loss"] += loss.item() * len(target)
            sums["correct"] += (logits.argmax(-1) == target.argmax(-1)).sum().item()
            seen += len(target)
        student.vit.set_record(False)
        for attention in teacher_layers:
            attention.record = False
            attention.cache = {}
        scores = evaluate(student, front, loaders["validation"], device)
        train_acc_eval = evaluate(student, front, train_eval, device)["top1_accuracy"]
        gap = max(0.0, train_acc_eval - scores["top1_accuracy"])
        score = scores["macro_acc"] - args.gap_penalty * gap
        row = {"epoch": epoch, "train_loss": sums["loss"] / seen, "train_acc": sums["correct"] / seen, "seconds": time.time() - started,
               "macro_acc": scores["macro_acc"], "train_acc_eval": train_acc_eval, "gap": gap, "score": score,
               "kd_qkv": sums["qkv"] / seen, "kd_scale": sums["scale"] / seen, "kd_pos": sums["pos"] / seen, **{k: scores[k] for k in REPORT}}
        row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        if cosine is None:
            if scores["macro_acc"] > lr_best:
                lr_best, lr_stale = scores["macro_acc"], 0
            else:
                lr_stale += 1
                if lr_stale >= args.lr_patience:
                    base_lr, lr_stale = max(base_lr * 0.5, 1e-6), 0
        if scores["macro_acc"] > best_val:
            best_val, best_val_epoch, stale = scores["macro_acc"], epoch, 0
            torch.save({"state_dict": copy.deepcopy(student.state_dict()), "epoch": epoch}, out_dir / "best_val.pt")
        else:
            stale += 1
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"state_dict": copy.deepcopy(student.state_dict()), "epoch": epoch}, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "params_m": params, "history": history}, indent=2))
        print("EPOCH " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())
              + f" | best_score {best:.4f}@{best_epoch} best_val {best_val:.4f}@{best_val_epoch} stale {stale}", flush=True)
        if stale >= args.patience:
            print(f"EARLY_STOP no val gain for {args.patience} epochs", flush=True)
            break
    chosen = next(r for r in history if r["epoch"] == best_epoch)
    print(f"DONE {args.name} chosen epoch {best_epoch}: val macro_acc {chosen['macro_acc']:.4f} mAP {chosen['mAP']:.4f} gap {chosen['gap']:.4f} "
          f"score {best:.4f} | best val {best_val:.4f}@{best_val_epoch}", flush=True)


if __name__ == "__main__":
    main()
