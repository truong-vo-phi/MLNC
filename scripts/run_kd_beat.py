"""KD teacher BEATs (nhận nhiễu thật) -> student FRCRN (đóng băng) + PANNs CNN10 bằng ReviewKD.

Mã từ notebook kd-beat-cnn10 được đưa vào module để chạy nền trên molab. Đánh giá giữ như cũ: balanced accuracy từng mốc SNR
(-5, 0, 5, 10, 15, 20) và trung bình `macro_acc` (cũng là tiêu chí chọn checkpoint).

    python -m scripts.run_kd_beat --smoke
    python -m scripts.run_kd_beat --use-drive --drive-dir FA26_Capstone/Truong/kd-beat-cnn10
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.drive_sync import DriveCheckpointUploader, DriveFileBackup  # noqa: E402
from scripts.finetune import METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES  # noqa: E402
from scripts.load_dataset import MIXTURE_KEY, Mixture36Dataset, build_dataloader, subsample  # noqa: E402
from scripts.load_pretrain import download_checkpoint, resolve_url  # noqa: E402
from scripts.mixdataset_speech import CleanReferenceDataset  # noqa: E402
from scripts.run_baselines import best_epoch_scores, count_millions  # noqa: E402
from scripts.run_reviewkd import (  # noqa: E402
    build_review,
    build_student,
    check_grids,
    read_reviewkd_config,
    resolve_path,
)
from src.model.baseline import BaselineModel  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.model.reviewkd import ReviewKDConfig, rms_normalize, tokens_to_grid  # noqa: E402
from src.train import ReviewKDTrainConfig, ReviewKDTrainer  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.paths import RunPaths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "reviewkd.yaml"  # student PANNs CNN10
DEFAULT_DRIVE_DIR = "FA26_Capstone/Truong/kd-beat-cnn10"
SNR_LEVELS = (-5, 0, 5, 10, 15, 20)
SNR_METRICS = tuple(f"balanced_snr_{snr}" for snr in SNR_LEVELS) + ("macro_acc",)
FRCRN_CONFIG = ROOT / "config" / "frcrn.yaml"
BEATS_CONFIG = ROOT / "config" / "beat.yaml"


class FRCRNStudent(nn.Module):
    """mixture -> FRCRN (đóng băng) -> n_hat = x - s_hat -> BaselineModel (log-mel + backbone)."""

    def __init__(self, separator: FRCRNSeparator, student: BaselineModel) -> None:
        super().__init__()
        self.separator = separator.freeze()
        self.student = student
        self.config = student.config

    def train(self, mode: bool = True) -> FRCRNStudent:
        super().train(mode)
        self.separator.eval()
        return self

    def forward(self, waveform: Tensor, return_features: bool = False):
        with torch.no_grad():
            noise_hat = self.separator(waveform)  # FRCRNSeparator.forward đã trả mixture - speech_hat
        return self.student(noise_hat, return_features=return_features)

    def predict_proba(self, waveform: Tensor) -> Tensor:
        return self(waveform).logits.softmax(dim=-1)


class NoiseTeacher(nn.Module):
    """nhiễu thật -> chuẩn hoá RMS -> BEATs -> 4 lưới (4, B, 768, 8, 24). Luôn eval, no_grad."""

    def __init__(self, beats: BEATClassifier, layers: tuple[int, ...], target_rms: float) -> None:
        super().__init__()
        if target_rms <= 0:
            raise ValueError(f"target_rms must be positive, got {target_rms}")
        self.beats = beats.freeze()
        self.layers = tuple(layers)
        self.target_rms = target_rms
        self.freq_bins = beats.frontend.num_mel_bins // beats.config.input_patch_size

    def train(self, mode: bool = True) -> NoiseTeacher:
        return super().train(False)

    @torch.no_grad()
    def forward(self, noise: Tensor, with_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        output = self.beats(rms_normalize(noise, self.target_rms), layers=self.layers)
        grids = torch.stack([tokens_to_grid(hidden, self.freq_bins) for hidden in output.hidden])
        return (grids, output.logits) if with_logits else grids


class NoiseKDTrainer(ReviewKDTrainer):
    """Teacher ăn nhiễu thật của batch; chỉ cập nhật tham số requires_grad."""

    @property
    def trainable_parameters(self):
        parameters = itertools.chain(self.model.parameters(), self.review.parameters())
        return (p for p in parameters if p.requires_grad)

    def training_step(self, batch):
        self._noise = batch["noise"].to(self.device)
        return super().training_step(batch)

    def _call_teacher(self, waveform):
        if self.config.logit_weight > 0:
            grids, self._teacher_logits = self.teacher(self._noise, with_logits=True)
            return grids
        return self.teacher(self._noise)


def ensure_beats_checkpoint(raw: dict[str, Any]) -> Path:
    path = resolve_path(raw["teacher"]["beats_checkpoint"])
    return path if path.is_file() else download_checkpoint(resolve_url("beats-pretrain"), path.parent)


def build_noise_teacher(raw: dict[str, Any], review: ReviewKDConfig, pretrain: bool, device: torch.device) -> NoiseTeacher:
    beats = BEATClassifier.from_yaml(BEATS_CONFIG)
    finetuned = raw["teacher"].get("beats_finetuned")
    if pretrain and not finetuned:  # bản fine-tune ghi đè toàn bộ trọng số (load_state_dict strict), không cần tải BEATs gốc
        beats.load_pretrained(ensure_beats_checkpoint(raw))
    if finetuned:
        load_checkpoint(resolve_path(finetuned), beats)  # BEATs fine-tune 36 lớp trên nhiễu (scripts.run_beats_noise)
    return NoiseTeacher(beats, review.teacher_layers, review.target_rms).to(device)


def build_frcrn_student(
    raw: dict[str, Any], use_frcrn: bool, frcrn_checkpoint: Path | None = None, load_pretrained: bool = True
) -> nn.Module:
    student = build_student(raw, load_pretrained)
    if not use_frcrn:
        return student
    separator = FRCRNSeparator.from_yaml(FRCRN_CONFIG)
    separator.load_pretrained(frcrn_checkpoint or resolve_path(raw["teacher"]["frcrn_checkpoint"]))
    return FRCRNStudent(separator, student)


def per_snr_table(scores: dict[str, float], levels: Sequence[int] = SNR_LEVELS) -> dict[str, dict[str, float]]:
    """Bảng 9 chỉ số x từng mức SNR (+ dòng `mean` trung bình các mức) từ khoá `<chỉ số>_snr_<mức>` của `snr_scores`."""
    table = {f"{level}": {name: scores[f"{name}_snr_{level}"] for name in METRIC_NAMES} for level in levels}
    table["mean"] = {name: sum(table[f"{level}"][name] for level in levels) / len(levels) for name in METRIC_NAMES}
    return table


def write_per_snr_csv(table: dict[str, dict[str, float]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", *METRIC_NAMES])
        for snr, row in table.items():
            writer.writerow([snr, *(f"{row[name]:.6f}" for name in METRIC_NAMES)])


def format_table(table: dict[str, dict[str, float]]) -> str:
    header = f"{'SNR':>5} " + " ".join(f"{name[:9]:>9}" for name in METRIC_NAMES)
    lines = [header] + [f"{snr:>5} " + " ".join(f"{row[name]:9.4f}" for name in METRIC_NAMES) for snr, row in table.items()]
    return chr(10).join(lines)


def noise_dataset_class(root: Path, split: str = "train") -> str:
    """'file' nếu manifest có tệp nhiễu thật, 'residual' nếu phải dùng nhiễu = mixture - clean."""
    import csv

    with open(root / split / "manifest.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return "file" if row.get("noise_path") and (root / row["noise_path"]).is_file() else "residual"


def build_loaders(raw: dict[str, Any], config: ReviewKDTrainConfig, max_samples: int | None, workers: int, logger: logging.Logger):
    duration = raw["dataset"]["duration"]
    probe = Mixture36Dataset(split="validation", duration=duration)
    source = noise_dataset_class(probe.root)
    logger.info("nhiễu tham chiếu cho teacher: %s", "tệp noise_path" if source == "file" else "mixture - clean")
    if source == "file":
        train_ds = Mixture36Dataset(split="train", duration=duration, load_noise=True)
    else:
        train_ds = CleanReferenceDataset(split="train", duration=duration, random_crop=False)
    return {
        "train": build_dataloader(subsample(train_ds, max_samples), config.batch_size, True, workers),
        "validation": build_dataloader(subsample(probe, max_samples), config.eval_batch_size, False, workers),
    }


def run_experiment(
    raw: dict[str, Any],
    name: str,
    run: RunPaths,
    device: torch.device,
    logger: logging.Logger,
    *,
    use_frcrn: bool = True,
    pretrain_teacher: bool = True,
    review_weight: float | None = None,
    logit_weight: float | None = None,
    logit_temperature: float | None = None,
    smoke: bool = False,
    max_samples: int | None = None,
    workers: int = 8,
    epochs: int | None = None,
    patience: int | None = None,
    frcrn_checkpoint: Path | None = None,
    on_checkpoint=None,
    resume_path: Path | None = None,
    drive_backup=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = check_grids(raw)
    review_config = ReviewKDConfig.from_dict(raw["review"])
    train_config = ReviewKDTrainConfig.from_dict(raw["train"])
    overrides = {
        k: v
        for k, v in (
            ("review_weight", review_weight),
            ("logit_weight", logit_weight),
            ("logit_temperature", logit_temperature),
            ("epochs", epochs),
            ("patience", patience),
        )
        if v is not None
    }
    train_config = replace(train_config, **overrides)
    if smoke:
        train_config = replace(
            train_config, epochs=1, patience=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH, review_warmup_epochs=0
        )
        max_samples, workers = SMOKE_SAMPLES, 0

    set_seed(int(raw["train"].get("seed", 42)))
    loaders = build_loaders(raw, train_config, max_samples, workers, logger)
    resuming = resume_path is not None and resume_path.is_file()  # resume.pt ghi đè trọng số student nên không cần nạp PANNs
    model = build_frcrn_student(raw, use_frcrn, frcrn_checkpoint, load_pretrained=not resuming).to(device)
    review = build_review(raw, review_config, grid)
    teacher = build_noise_teacher(raw, review_config, pretrain_teacher, device) if train_config.review_weight > 0 else None
    checkpoint_path = run.checkpoint_path(name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "%s | arch=%s | frcrn=%s | teacher pretrain=%s | lambda=%.3f | model %.2fM | review %.2fM",
        name, raw["model"]["arch"], use_frcrn, pretrain_teacher, train_config.review_weight,
        count_millions(model), count_millions(review),
    )
    trainer = NoiseKDTrainer(
        model, review, teacher, train_config, device, logger, name, checkpoint_path, on_checkpoint, resume_path, drive_backup
    )
    if resume_path is not None and resume_path.is_file():
        trainer.load_resume(resume_path)
    set_seed(int(raw["train"].get("seed", 42)))
    fit = trainer.fit(loaders["train"], loaders["validation"])
    scores = best_epoch_scores(fit.history, fit.best_epoch)
    val_table = per_snr_table(scores)
    write_per_snr_csv(val_table, run.result_dir / f"per_snr_validation_{name}.csv")
    logger.info("%s: validation (best epoch %d), 9 metrics per SNR%s%s", name, fit.best_epoch, chr(10), format_table(val_table))
    test_table = None
    if not smoke:
        test_loader = build_dataloader(
            Mixture36Dataset(split="test", duration=raw["dataset"]["duration"]), train_config.eval_batch_size, False, workers
        )
        test_scores = trainer.evaluate(test_loader, desc=f"{name} - test")  # trainer.fit đã nạp lại trọng số tốt nhất
        test_table = per_snr_table(test_scores)
        write_per_snr_csv(test_table, run.result_dir / f"per_snr_test_{name}.csv")
        logger.info("%s: TEST (best epoch %d), 9 metrics per SNR%s%s", name, fit.best_epoch, chr(10), format_table(test_table))
    row = {
        "model": name,
        "arch": raw["model"]["arch"],
        "best_epoch": fit.best_epoch,
        "epochs_run": fit.epochs_run,
        **{metric: scores[metric] for metric in SNR_METRICS},
    }
    run.history_path(name).parent.mkdir(parents=True, exist_ok=True)
    run.history_path(name).write_text(
        json.dumps({"row": row, "history": fit.history, "per_snr_validation": val_table, "per_snr_test": test_table}, indent=2),
        encoding="utf-8",
    )
    logger.info("%s: validation | %s", name, " ".join(f"{k} {row[k]:.4f}" for k in SNR_METRICS))
    return row, fit.history


def experiment_name(
    arch: str,
    use_frcrn: bool,
    review_weight: float | None,
    pretrain_teacher: bool,
    finetuned_teacher: bool = False,
    logit_kd: bool = False,
) -> str:
    return "-".join([
        arch,
        "frcrn" if use_frcrn else "mix",
        "kd" if (review_weight is None or review_weight > 0) else "ce",
        "ft" if finetuned_teacher else ("pre" if pretrain_teacher else "rand"),
        *(["logit"] if logit_kd else []),
    ])


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", default="kd-beat")
    parser.add_argument("--no-frcrn", action="store_true", help="student nhận mixture trực tiếp (A0)")
    parser.add_argument("--random-teacher", action="store_true", help="BEATs khởi tạo ngẫu nhiên, đóng băng (B2)")
    parser.add_argument("--review-weight", type=float, default=None, help="0 = baseline CE thuần")
    parser.add_argument("--logit-weight", type=float, default=None, help="trọng số KD logit (0 = tắt); cần teacher đã fine-tune 36 lớp")
    parser.add_argument("--logit-temperature", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 64 mẫu, ghi vào output/test")
    parser.add_argument("--beats-finetuned", type=Path, default=None, help="best.pt của BEATs đã fine-tune 36 lớp (scripts.run_beats_noise), ghi đè teacher.beats_finetuned")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frcrn-checkpoint", type=Path, default=None, help="ghi đè teacher.frcrn_checkpoint")
    parser.add_argument("--frcrn-drive-path", default=None, help="tải checkpoint FRCRN từ Drive nếu chưa có trên máy")
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--drive-dir", default=DEFAULT_DRIVE_DIR)
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args(argv)

    raw = read_reviewkd_config(args.config)
    if args.beats_finetuned is not None:
        raw["teacher"]["beats_finetuned"] = str(args.beats_finetuned)
    base = args.work_dir / "test" if args.smoke else args.work_dir
    run = RunPaths.new(args.experiment + ("-smoke" if args.smoke else ""), base)
    run.create()
    logger = setup_logger(args.experiment, run.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run %s | device %s | config %s", run.run_id, device, args.config)

    logger.info("mixture-36l: %s", ensure_dataset(MIXTURE_KEY, args.download_root))
    fs = None
    if args.use_drive or args.frcrn_drive_path:
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, token="cache")
    frcrn_checkpoint = args.frcrn_checkpoint or resolve_path(raw["teacher"]["frcrn_checkpoint"])
    if not args.no_frcrn and not frcrn_checkpoint.is_file():
        if not args.frcrn_drive_path:
            raise FileNotFoundError(f"thiếu {frcrn_checkpoint}: truyền --frcrn-checkpoint hoặc --frcrn-drive-path")
        frcrn_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(args.frcrn_drive_path, str(frcrn_checkpoint))
        logger.info("FRCRN: %s -> %s", args.frcrn_drive_path, frcrn_checkpoint)

    use_frcrn, pretrain = not args.no_frcrn, not args.random_teacher
    name = experiment_name(raw["model"]["arch"], use_frcrn, args.review_weight, pretrain, bool(raw["teacher"].get("beats_finetuned")), bool(args.logit_weight))
    on_checkpoint = drive_backup = None
    remote_dir = f"{args.drive_dir.strip('/')}/{name}"
    resume_path = args.work_dir / "resume" / name / "resume.pt"  # cố định giữa các lần chạy để nối tiếp được
    if fs is not None and args.use_drive and not args.smoke:
        on_checkpoint = DriveCheckpointUploader(fs, remote_dir, logger)
        drive_backup = DriveFileBackup(fs, remote_dir, "resume.pt", logger)
        if not resume_path.is_file():
            try:
                if fs.exists(f"{remote_dir}/resume.pt"):
                    resume_path.parent.mkdir(parents=True, exist_ok=True)
                    fs.get_file(f"{remote_dir}/resume.pt", str(resume_path))
                    logger.info("drive: khôi phục %s/resume.pt", remote_dir)
            except Exception as error:
                logger.warning("drive: không khôi phục được resume.pt: %r", error)
    if args.smoke:
        resume_path = None
    row, _ = run_experiment(
        raw, name, run, device, logger,
        use_frcrn=use_frcrn, pretrain_teacher=pretrain, review_weight=args.review_weight, logit_weight=args.logit_weight, logit_temperature=args.logit_temperature, smoke=args.smoke,
        max_samples=args.max_samples, workers=args.workers, epochs=args.epochs, patience=args.patience,
        frcrn_checkpoint=frcrn_checkpoint, on_checkpoint=on_checkpoint, resume_path=resume_path, drive_backup=drive_backup,
    )
    if fs is not None and args.use_drive and not args.smoke:
        DriveFileBackup(fs, f"{args.drive_dir.strip('/')}/{name}", run.history_path(name).name, logger)(run.history_path(name))
    (run.result_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


if __name__ == "__main__":
    main()
