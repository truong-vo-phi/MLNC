"""Student là BEATs cắt còn L lớp, khởi tạo từ teacher (best.pt), tinh chỉnh bằng CE + KD logit từ teacher 12 lớp.

    python -m scripts.run_beats_trunc --noise-cache /marimo/cache/nhat --layers 6 --epochs 20 --patience 5

`best.pt` chọn theo điểm = macro_acc - penalty * chênh lệch train/val (train đo ở chế độ eval); `best_val.pt` là epoch có macro_acc cao nhất. Dừng sớm khi macro_acc validation không tăng sau `--patience` epoch. Không đụng tới test.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewkd_deit import BEATS_CONFIG, CONFIG, build_cached_loaders, load_teacher_checkpoint  # noqa: E402
from scripts.load_dataset import build_dataloader, subsample  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.beat.beat import BEATConfig  # noqa: E402
from src.train import ReviewKDTrainConfig  # noqa: E402
from src.train.beat_trainer import cosine_with_warmup  # noqa: E402
from src.train.reviewkd_trainer import snr_scores  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LAYER_KEY = re.compile(r"\.layers\.(\d+)\.")
NUM_CLASSES = 36
REPORT = ("top1_accuracy", "balanced_accuracy", "macro_f1", "mAP", "macro_auc")


FFN_KEY = re.compile(r"(backbone\.encoder\.layers\.(\d+)\.ffn\.)(fc1\.weight|fc1\.bias|fc2\.weight)$")


@torch.no_grad()
def ffn_importance(teacher: BEATClassifier, front, loader, device, layers: int, batches: int) -> list[torch.Tensor]:
    """Độ lớn trung bình của kích hoạt (sau GELU) từng nơ-ron FFN của `layers` lớp đầu trên một ít batch train."""
    totals = [None] * layers
    counts = [0] * layers

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            value = output.detach().float().abs().sum(dim=tuple(range(output.dim() - 1)))
            totals[index] = value if totals[index] is None else totals[index] + value
            counts[index] += output[..., 0].numel()

        return hook

    handles = [teacher.backbone.encoder.layers[i].ffn.act.register_forward_hook(make_hook(i)) for i in range(layers)]
    teacher.eval()
    for step, batch in enumerate(loader):
        if step >= batches:
            break
        teacher(front(batch["waveform"].to(device)))
    for handle in handles:
        handle.remove()
    return [total / count for total, count in zip(totals, counts, strict=True)]


def build_student(teacher: BEATClassifier, layers: int, ffn_dim: int | None = None, importance: list[torch.Tensor] | None = None) -> BEATClassifier:
    raw = yaml.safe_load(open(BEATS_CONFIG, encoding="utf-8"))
    config = replace(BEATConfig.from_dict(raw["model"]), encoder_layers=layers)
    if ffn_dim is not None:
        config = replace(config, encoder_ffn_embed_dim=ffn_dim)
    student = BEATClassifier(config, raw["frontend"])
    state = {k: v for k, v in teacher.state_dict().items() if not (m := LAYER_KEY.search(k)) or int(m[1]) < layers}
    if ffn_dim is not None:
        keep = [torch.topk(score, ffn_dim).indices.sort().values.cpu() for score in importance]
        for key in list(state):
            m = FFN_KEY.search(key)
            if m:
                index = keep[int(m[2])]
                state[key] = state[key][:, index] if m[3] == "fc2.weight" else state[key][index]
    result = student.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"student init: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
    return student


@torch.no_grad()
def evaluate(student: BEATClassifier, front, loader, device) -> dict[str, float]:
    student.eval()
    probabilities, labels, snrs = [], [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = student(front(batch["waveform"].to(device))).logits
        probabilities.append(logits.float().softmax(-1).cpu())
        labels.append(batch["target"].argmax(dim=1))
        snrs.append(batch["snr_db"])
    return snr_scores(torch.cat(probabilities).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), NUM_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=None, help="cắt FFN từ 3072 xuống số nơ-ron này (giữ nơ-ron kích hoạt mạnh nhất)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logit-weight", type=float, default=1.0, help="0 = chỉ CE")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0, help="xác suất mixup mỗi batch (waveform n^, nhãn mềm)")
    parser.add_argument("--gap-penalty", type=float, default=0.5, help="điểm chọn = macro_acc - penalty * max(0, train_acc_eval - val_acc)")
    parser.add_argument("--train-eval-clips", type=int, default=2000, help="số clip train (cố định) để đo train acc ở chế độ eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "beats-trunc")
    args = parser.parse_args()

    raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    config = ReviewKDTrainConfig.from_dict({**raw["train"], "batch_size": args.batch_size})
    loaders = build_cached_loaders(raw, config, args.noise_cache, args.max_samples, 4)
    front, teacher, _ = load_teacher_checkpoint(ROOT / raw["teacher"]["checkpoint"], device, use_frcrn=False)
    teacher.freeze()
    train_eval = build_dataloader(subsample(loaders["train"].dataset, args.train_eval_clips), 64, False, 4)
    importance = ffn_importance(teacher, front, loaders["train"], device, args.layers, 16) if args.ffn_dim else None
    student = build_student(teacher, args.layers, args.ffn_dim, importance).to(device)
    name = args.name or f"beats{args.layers}L-ffn{args.ffn_dim or 3072}-kd{args.logit_weight:g}-mix{args.mixup:g}-lr{args.lr:g}"
    out_dir = args.output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"RUN {name} | student {params:.1f}M params | {args.layers} layers | args {vars(args)}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    steps = args.epochs * len(loaders["train"])
    scheduler = cosine_with_warmup(optimizer, steps, 0.1, 0.0)
    best, best_epoch, stale, best_state, history = -1e9, 0, 0, None, []
    best_val, best_val_epoch, best_val_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        student.train()
        started, loss_sum, correct, seen = time.time(), 0.0, 0, 0
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
                logits = student(wave).logits.float()
                smooth = target * (1 - args.label_smoothing) + args.label_smoothing / NUM_CLASSES
                loss = -(smooth * F.log_softmax(logits, -1)).sum(-1).mean()
                if args.logit_weight > 0:
                    with torch.no_grad():
                        teacher_logits = teacher(wave).logits.float()
                    kd = F.kl_div(F.log_softmax(logits / args.temperature, -1), F.softmax(teacher_logits / args.temperature, -1), reduction="batchmean")
                    loss = loss + args.logit_weight * kd * args.temperature**2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(target)
            correct += (logits.argmax(-1) == target.argmax(-1)).sum().item()
            seen += len(target)
        scores = evaluate(student, front, loaders["validation"], device)
        train_acc_eval = evaluate(student, front, train_eval, device)["top1_accuracy"]
        gap = max(0.0, train_acc_eval - scores["top1_accuracy"])
        score = scores["macro_acc"] - args.gap_penalty * gap
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen, "seconds": time.time() - started,
               "macro_acc": scores["macro_acc"], "train_acc_eval": train_acc_eval, "gap": gap, "score": score, **{k: scores[k] for k in REPORT}}
        history.append(row)
        if scores["macro_acc"] > best_val:
            best_val, best_val_epoch, stale = scores["macro_acc"], epoch, 0
            best_val_state = copy.deepcopy(student.state_dict())
        else:
            stale += 1
        if score > best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(student.state_dict())
        print("EPOCH " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()) + f" | best_score {best:.4f}@{best_epoch} best_val {best_val:.4f}@{best_val_epoch} stale {stale}", flush=True)
        if stale >= args.patience:
            print(f"EARLY_STOP no val gain for {args.patience} epochs", flush=True)
            break
    torch.save({"state_dict": best_state, "layers": args.layers, "ffn_dim": args.ffn_dim, "best_epoch": best_epoch}, out_dir / "best.pt")
    torch.save({"state_dict": best_val_state, "layers": args.layers, "ffn_dim": args.ffn_dim, "best_epoch": best_val_epoch}, out_dir / "best_val.pt")
    (out_dir / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "params_m": params, "history": history}, indent=2))
    chosen = next(r for r in history if r["epoch"] == best_epoch)
    print(f"DONE {name} chosen epoch {best_epoch}: val macro_acc {chosen['macro_acc']:.4f} mAP {chosen['mAP']:.4f} gap {chosen['gap']:.4f} score {best:.4f} | best val {best_val:.4f}@{best_val_epoch}", flush=True)


if __name__ == "__main__":
    main()
