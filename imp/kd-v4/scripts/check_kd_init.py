"""Kiểm tra trước khi huấn luyện: ViTSBias chạy đúng, hoán vị token, ghép lớp/head, và loss KD hữu hạn.

    python -m scripts.check_kd_init --noise-cache /marimo/cache/nhat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewkd_deit import CONFIG, build_cached_loaders, load_teacher_checkpoint  # noqa: E402
from scripts.run_vits_kd import Student, teacher_attentions  # noqa: E402
from src.loss.beat_attention_kd import AttentionKD, group_heads  # noqa: E402
from src.train import ReviewKDTrainConfig  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-cache", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ReviewKDTrainConfig.from_dict({**raw["train"], "batch_size": 8})
    loaders = build_cached_loaders(raw, config, args.noise_cache, 32, 0)
    front, teacher, _ = load_teacher_checkpoint(ROOT / raw["teacher"]["checkpoint"], device, use_frcrn=False)
    teacher.freeze()
    model = raw["model"]
    specaug = dict(freq_masks=model["specaug_freq_masks"], freq_width=model["specaug_freq_width"],
                   time_masks=model["specaug_time_masks"], time_width=model["specaug_time_width"])
    student = Student(raw["frontend"], specaug, num_classes=36, pretrained=False).to(device).eval()
    teacher_layers = teacher_attentions(teacher)
    table = teacher_layers[0].relative_attention_bias.weight.detach().cpu()
    student.vit.init_bias_from_teacher(table)

    batch = next(iter(loaders["validation"]))
    wave = front(batch["waveform"].to(device))
    student.vit.set_record(True)
    for attention in teacher_layers:
        attention.record = True
    logits = student(wave)
    teacher_logits = teacher(wave).logits
    s_cache = [a.cache for a in student.vit.attentions]
    t_cache = [a.cache for a in teacher_layers]
    print("student logits", tuple(logits.shape), "teacher logits", tuple(teacher_logits.shape), "finite", bool(torch.isfinite(logits).all()))
    print("student q", tuple(s_cache[0]["q"].shape), "teacher q", tuple(t_cache[0]["q"].shape))
    print("student P", tuple(s_cache[0]["position_bias"].shape), "teacher P", tuple(t_cache[0]["position_bias"].shape))
    heads = s_cache[0]["q"].shape[1]
    p_gap = (s_cache[0]["position_bias"][:, :, 1:, 1:].float() - group_heads(t_cache[0]["position_bias"].float(), heads)).abs().max().item()
    print(f"max |P_student - P_teacher(grouped)| = {p_gap:.2e}  (phải gần 0)")
    terms = AttentionKD(1.0, 1.0, 1.0)(s_cache, t_cache)
    print("KD terms at init:", {k: round(v.item(), 4) for k, v in terms.items()}, "finite", all(bool(torch.isfinite(v)) for v in terms.values()))
    ok = p_gap < 1e-4 and bool(torch.isfinite(logits).all()) and all(bool(torch.isfinite(v)) for v in terms.values())
    print("CHECK", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
