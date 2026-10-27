"""ReviewKD cho DeiT-Tiny: FRCRN chạy một lần mỗi batch, n^ đi vào cả teacher BEATs và student DeiT.

    python -m scripts.run_reviewkd_deit [--smoke] [--epochs N] [--max-samples N] [--run-id ID] [--review-weight W]

Teacher lấy từ một tệp `best.pt` có BEATs fine-tune và FRCRN đi kèm (format `beats_mlkd_frcrn_v1`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune import METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES  # noqa: E402
from scripts.load_dataset import build_dataloader, build_dataset, subsample  # noqa: E402
from scripts.run_baselines import best_epoch_scores, count_millions, write_result_csv  # noqa: E402
from scripts.run_reviewkd import RESULT_COLUMNS, teacher_grid  # noqa: E402
from src.model.baseline import BaselineConfig, BaselineModel  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.model.reviewkd import ReviewKDConfig, ReviewModule, rms_normalize, tokens_to_grid  # noqa: E402
from src.train import ReviewKDTrainConfig, ReviewKDTrainer  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.paths import RunPaths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

CONFIG = ROOT / "config" / "reviewkd_deit.yaml"
FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"
BEATS_CONFIG = ROOT / "config" / "beat.yaml"
EXPECTED_FORMAT = "beats_mlkd_frcrn_v1"
ENCODER_PREFIX = "encoder."
HEAD_PREFIX = "head."
FRCRN_PREFIX = "frcrn."


class BlockwiseHCL(nn.Module):
    """Block-wise KD: loss của khối thứ k (đặc trưng khối 3k của DeiT so với lớp 3k của BEATs) chỉ bật từ epoch `unlock[k]`.

    Các khối đầu học khớp teacher trước, khối sau được mở dần; CE trên logit vẫn chạy suốt.
    """

    def __init__(self, inner: nn.Module, unlock: tuple[int, ...]) -> None:
        super().__init__()
        self.inner, self.unlock, self.epoch = inner, unlock, 1

    def per_level(self, student: Tensor, teacher: Tensor) -> Tensor:
        levels = self.inner.per_level(student, teacher)
        if len(self.unlock) != len(levels):
            raise ValueError(f"{len(self.unlock)} unlock epochs for {len(levels)} stages")
        active = torch.tensor([float(self.epoch >= epoch) for epoch in self.unlock], device=levels.device)
        return levels * active


class NoiseCacheDataset(Dataset):
    """Thay `waveform` bằng n^ đã tính trước (memmap float32, cùng thứ tự với manifest), nên không phải chạy FRCRN."""

    def __init__(self, base: Dataset, path: Path) -> None:
        self.base, self.path, self._array = base, path, None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        item = dict(self.base[index])
        item["waveform"] = torch.from_numpy(np.array(self._array[index]))
        return item


def build_cached_loaders(raw: dict[str, Any], config: ReviewKDTrainConfig, cache_dir: Path | None, max_samples, num_workers) -> dict[str, Any]:
    name, duration = raw["dataset"]["name"], raw["dataset"]["duration"]
    loaders: dict[str, Any] = {}
    for split, batch_size, shuffle in (("train", config.batch_size, True), ("validation", config.eval_batch_size, False)):
        dataset = build_dataset(name, split, duration=duration)
        if cache_dir is not None:
            array = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
            if len(array) != len(dataset):
                raise ValueError(f"{cache_dir / f'{split}.npy'}: {len(array)} rows but the {split} manifest has {len(dataset)}")
            if (cache_dir / f"{split}.done").read_text().strip() != str(len(dataset)):
                raise ValueError(f"{split}: cache is not marked complete")
            dataset = NoiseCacheDataset(dataset, cache_dir / f"{split}.npy")
        loaders[split] = build_dataloader(subsample(dataset, max_samples), batch_size, shuffle, num_workers)
        if split == "train":
            loaders["class_names"] = build_dataset(name, split, duration=duration).class_names
    return loaders


class SharedFront(nn.Module):
    """FRCRN đóng băng + chuẩn hoá RMS; giữ n^ của lần gọi gần nhất để teacher dùng lại.

    `separator=None`: đầu vào đã là n^ (từ cache) nên chỉ chuẩn hoá RMS.
    """

    def __init__(self, separator: FRCRNSeparator | None, input_rms: float) -> None:
        super().__init__()
        self.separator = separator.freeze() if separator is not None else None
        self.input_rms = input_rms
        self.noise: Tensor | None = None

    def train(self, mode: bool = True) -> SharedFront:
        return super().train(False)

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        estimate = waveform if self.separator is None else self.separator(waveform)
        self.noise = rms_normalize(estimate, self.input_rms)
        return self.noise


class FrontedStudent(nn.Module):
    """Student nhìn n^ thay vì mixture. state_dict chỉ gồm student để checkpoint không mang theo FRCRN."""

    def __init__(self, front: SharedFront, student: BaselineModel) -> None:
        super().__init__()
        self.front = front
        self.student = student
        self.config = student.config

    def forward(self, waveform: Tensor, return_features: bool = False):
        return self.student(self.front(waveform), return_features=return_features)

    def predict_proba(self, waveform: Tensor) -> Tensor:
        return self.student.predict_proba(self.front(waveform))

    def parameters(self, recurse: bool = True):
        return self.student.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.student.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        return self.student.load_state_dict(state_dict, strict=strict, **kwargs)


class FrontedTeacher(nn.Module):
    """BEATs đóng băng đọc n^ mà student vừa dùng: phải gọi sau FrontedStudent trong cùng batch."""

    def __init__(self, front: SharedFront, beats: BEATClassifier, layers: tuple[int, ...]) -> None:
        super().__init__()
        self.front = front
        self.beats = beats.freeze()
        self.layers = layers
        self.freq_bins = beats.frontend.num_mel_bins // beats.config.input_patch_size

    def train(self, mode: bool = True) -> FrontedTeacher:
        return super().train(False)

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        if self.front.noise is None or self.front.noise.shape[0] != waveform.shape[0]:
            raise RuntimeError("the student must run on this batch before the teacher")
        output = self.beats(self.front.noise, layers=self.layers)
        return torch.stack([tokens_to_grid(hidden, self.freq_bins) for hidden in output.hidden])


FOLD_PREFIX = "encoder.model."
FOLD_INPUT_RMS = 0.07925  # chuẩn hoá RMS nhiễu trước fbank, như config/reviewkd.yaml và kd-v3


def load_fold_teacher(checkpoint: dict[str, Any], device: torch.device) -> tuple[SharedFront, BEATClassifier, dict[str, Any]]:
    """`beats_teacher_fold*_weights.pt`: BEATs fine-tune 36 lớp trên nhiễu thật (khoá `encoder.model.*`, `head.*`), không kèm FRCRN."""
    state = checkpoint["state_dict"]
    backbone = {k[len(FOLD_PREFIX) :]: v for k, v in state.items() if k.startswith(FOLD_PREFIX)}
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    converted = beats.convert_checkpoint({"model": backbone})
    converted["head.fc.weight"], converted["head.fc.bias"] = state["head.weight"], state["head.bias"]
    result = beats.load_state_dict(converted, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"BEATs fold: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
    meta = {"labels": None, "input_rms": FOLD_INPUT_RMS, "epoch": checkpoint.get("best_epoch")}
    return SharedFront(None, FOLD_INPUT_RMS).to(device), beats.to(device), meta


def load_teacher_checkpoint(path: Path, device: torch.device, use_frcrn: bool = True) -> tuple[SharedFront, BEATClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "format" not in checkpoint and any(k.startswith(FOLD_PREFIX) for k in checkpoint.get("state_dict", {})):
        if use_frcrn:
            raise ValueError(f"{path} is a fold checkpoint without FRCRN: pass a noise cache (use_frcrn=False)")
        return load_fold_teacher(checkpoint, device)
    if checkpoint.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"{path}: format {checkpoint.get('format')!r}, expected {EXPECTED_FORMAT!r}")
    state = checkpoint["state_dict"]

    separator = None
    if use_frcrn:
        base = FRCRNSeparator.from_yaml(FRCRN_CONFIG)
        separator = FRCRNSeparator(replace(base.config, **checkpoint["frcrn_config"]))
        frcrn_state = {k[len(FRCRN_PREFIX) :]: v for k, v in state.items() if k.startswith(FRCRN_PREFIX)}
        result = separator.load_state_dict(frcrn_state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"FRCRN: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
        separator = separator.to(device)

    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    original = {k[len(ENCODER_PREFIX) :]: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
    converted = beats.convert_checkpoint({"model": original})
    converted.update({f"head.fc.{k[len(HEAD_PREFIX) :]}": v for k, v in state.items() if k.startswith(HEAD_PREFIX)})
    result = beats.load_state_dict(converted, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"BEATs: missing {result.missing_keys[:3]}, unexpected {result.unexpected_keys[:3]}")
    meta = {key: checkpoint[key] for key in ("labels", "input_rms", "epoch")}
    return SharedFront(separator, float(checkpoint["input_rms"])).to(device), beats.to(device), meta


@torch.no_grad()
def probe_teacher(front: SharedFront, beats: BEATClassifier, loader, device: torch.device) -> float:
    """Độ chính xác top-1 của teacher trên n^ cho một phần validation (kiểm tra teacher trước khi chưng cất)."""
    correct = seen = 0
    for batch in loader:
        waveform = batch["waveform"].to(device)
        labels = batch["target"].argmax(dim=1).to(device)
        logits = beats(front(waveform)).logits
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        seen += len(labels)
    return correct / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, ít mẫu, chỉ kiểm tra đường chạy")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--review-weight", type=float, default=None, help="0 = baseline CE (vẫn qua FRCRN)")
    parser.add_argument("--run-id", default=None, help="nối tiếp một lần chạy cũ, dạng <tên>-YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--blockwise-unlock", default=None, help="block-wise KD: epoch bật loss của từng khối, ví dụ 1,4,7,10")
    parser.add_argument("--noise-cache", type=Path, default=None, help="thư mục có train.npy/validation.npy chứa n^ đã tính; bỏ qua FRCRN")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    run = RunPaths.from_run_id(args.run_id, args.output_dir) if args.run_id else RunPaths.new("reviewkd-deit", args.output_dir)
    run.create()
    logger = setup_logger("reviewkd-deit", run.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run %s | device %s | config %s | noise cache %s", run.run_id, device, args.config, args.noise_cache)

    train_config = ReviewKDTrainConfig.from_dict(raw["train"])
    if args.epochs is not None:
        train_config = replace(train_config, epochs=args.epochs, patience=max(train_config.patience, args.epochs))
    if args.review_weight is not None:
        train_config = replace(train_config, review_weight=args.review_weight)
    max_samples, num_workers = args.max_samples, args.num_workers
    if args.smoke:
        train_config = replace(
            train_config, epochs=1, patience=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH, review_warmup_epochs=0
        )
        max_samples, num_workers = SMOKE_SAMPLES, 0

    unlock = tuple(int(e) for e in args.blockwise_unlock.split(",")) if args.blockwise_unlock else None
    name = "deit-ce" if train_config.review_weight <= 0 else ("deit-blockwise" if unlock else "deit-reviewkd")
    history_path = run.history_path(name)
    if history_path.is_file():
        logger.info("%s: finished earlier, see %s", name, history_path)
        return

    set_seed(int(raw["train"].get("seed", 42)))
    loaders = build_cached_loaders(raw, train_config, args.noise_cache, max_samples, num_workers)

    front, beats, meta = load_teacher_checkpoint(ROOT / raw["teacher"]["checkpoint"], device, use_frcrn=args.noise_cache is None)
    if meta["labels"] is not None and list(meta["labels"]) != list(loaders["class_names"]):
        raise ValueError("teacher labels differ from the dataset class order")
    probe_set = loaders["validation"].dataset.dataset if hasattr(loaders["validation"].dataset, "dataset") else loaders["validation"].dataset
    probe = build_dataloader(subsample(probe_set, raw["teacher"]["probe_clips"]), train_config.eval_batch_size, False, num_workers)
    teacher_acc = probe_teacher(front, beats, probe, device)
    logger.info("teacher | epoch %s | input_rms %.4f | top-1 on n^ (%d val clips) %.4f",
                meta["epoch"], meta["input_rms"], raw["teacher"]["probe_clips"], teacher_acc)

    review_config = ReviewKDConfig.from_dict(raw["review"])
    grid = teacher_grid(raw["dataset"]["duration"], raw["frontend"])
    student = BaselineModel(BaselineConfig.from_dict(raw["model"]), raw["frontend"])
    embed_dim = student.backbone.network.embed_dim
    review_module = ReviewModule([embed_dim] * 4, review_config.teacher_dim, grid, review_config.mid_channels, review_config.resize)
    model = FrontedStudent(front, student)
    teacher = FrontedTeacher(front, beats, tuple(review_config.teacher_layers)) if train_config.review_weight > 0 else None

    checkpoint_path = run.checkpoint_path(name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_path.parent / "resume.pt"
    params_m = count_millions(student)
    logger.info("%s | student %.2fM params | review module %.2fM | grid %s | lambda %.3f | epochs %d",
                name, params_m, count_millions(review_module), grid, train_config.review_weight, train_config.epochs)

    trainer = ReviewKDTrainer(model, review_module, teacher, train_config, device, logger, name, checkpoint_path, resume_path=resume_path)
    if unlock:
        trainer.hcl = BlockwiseHCL(trainer.hcl, unlock)
        run_epoch = trainer.run_epoch

        def run_epoch_with_schedule(loader, epoch, _run=run_epoch):
            trainer.hcl.epoch = epoch
            return _run(loader, epoch)

        trainer.run_epoch = run_epoch_with_schedule
        logger.info("block-wise KD | khối mở ở epoch %s", unlock)
    if resume_path.is_file():
        trainer.load_resume(resume_path)
    set_seed(int(raw["train"].get("seed", 42)))
    started = time.time()
    fit = trainer.fit(loaders["train"], loaders["validation"])
    scores = best_epoch_scores(fit.history, fit.best_epoch)
    row = {
        "model": name, "review_weight": train_config.review_weight, "params_m": round(params_m, 2),
        "best_epoch": fit.best_epoch, "epochs_run": fit.epochs_run, "train_seconds": round(time.time() - started, 1),
        "macro_acc": scores["macro_acc"], **{metric: scores[metric] for metric in METRIC_NAMES},
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"row": row, "history": fit.history, "teacher_top1_probe": teacher_acc}, indent=2), encoding="utf-8")
    write_result_csv([row], run.result_file, RESULT_COLUMNS)
    logger.info("%s: validation | %s", name, " ".join(f"{k} {row[k]:.4f}" for k in ("macro_acc", *METRIC_NAMES)))


if __name__ == "__main__":
    main()
