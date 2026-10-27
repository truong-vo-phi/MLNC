"""Fine-tune BEATs (36 lớp) trên bộ nhiễu GỐC AudioSet 36L: clip nhiễu thuần, chuẩn hoá RMS, đúng kiểu đầu vào teacher nhận trong KD.

Train: clip AudioSet 40L (split train), bỏ clip mang 4 nhãn ngoài 36 lớp, bỏ clip có ytid nằm trong validation/test của Mixture-36l
(tránh rò rỉ sang đánh giá của student), mặc định chỉ giữ clip đúng một nhãn 36 lớp (Mixture-36l cũng chỉ có một nhãn mỗi clip).
Mỗi lần đọc cắt ngẫu nhiên 4 giây. Validation: noise_path của Mixture-36l (cùng phân phối với đầu vào teacher lúc KD).
Teacher này dùng cho logit KD và/hoặc feature KD (`teacher.beats_finetuned`). Tiêu chí chọn epoch là macro_f1 của `BEATTrainer`.

    python -m scripts.run_beats_noise --smoke
    python -m scripts.run_beats_noise --use-drive --drive-dir FA26_Capstone/Truong/beats-noise-ft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset  # noqa: E402
from scripts.dynamic_mix_36l import dropped_label_names, filter_noise_36l, mixture_class_names  # noqa: E402
from scripts.finetune import (  # noqa: E402
    BEAT_CONFIG,
    CLIP_SECONDS,
    METRIC_NAMES,
    SMOKE_BATCH,
    SMOKE_SAMPLES,
    prepare_stage,
    save_result,
    set_trainable,
)
from scripts.load_dataset import (  # noqa: E402
    AUDIOSET_KEY,
    MIXTURE_KEY,
    AudioSet40Dataset,
    Mixture36Dataset,
    build_dataloader,
    read_noise_ytids,
    subsample,
)
from scripts.load_pretrain import download_checkpoint, resolve_url  # noqa: E402
from scripts.run_reviewkd import read_reviewkd_config, resolve_path  # noqa: E402
from scripts.run_frcrn_noise import setup_logging  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.reviewkd import rms_normalize  # noqa: E402
from src.train import BEATTrainConfig, BEATTrainer  # noqa: E402
from src.train.reviewkd_trainer import snr_scores  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DEFAULT_DRIVE_DIR = "FA26_Capstone/Truong/beats-noise-ft"
STAGE_NAME = "beat_noise"

logger = logging.getLogger("run_beats_noise")


def noise_input(batch: dict[str, Any], target_rms: float, device: torch.device) -> tuple[Tensor, Tensor]:
    """Đầu vào teacher: nhiễu thật chuẩn hoá RMS, nhãn là lớp nhiễu của clip."""
    noise = batch["noise"].to(device)
    labels = batch["target"].argmax(dim=1).to(device)
    return rms_normalize(noise, target_rms), labels


def every_n_calls(function: Any, n: int) -> Any:
    """Gọi `function` ở lần gọi thứ n, 2n, ... (lần đầu bỏ qua) để giảm lượng upload resume.pt lớn."""
    count = 0

    def wrapper(*args: Any, **kwargs: Any) -> None:
        nonlocal count
        count += 1
        if count % n == 0:
            function(*args, **kwargs)

    return wrapper


class NoiseBEATTrainer(BEATTrainer):
    """BEATTrainer nhận nhiễu thật thay cho mixture (không có FRCRN)."""

    def __init__(self, *args: Any, target_rms: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_rms = target_rms

    def prepare(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        return noise_input(batch, self.target_rms, self.device)

    @torch.no_grad()
    def _evaluate_with(self, predict_proba: Any, loader: Any, desc: str) -> dict[str, float]:
        """9 chỉ số tổng, cộng 9 chỉ số theo từng mức SNR của clip (`<chỉ số>_snr_<mức>`, ví dụ `mAP_snr_-5`)."""
        probabilities, labels, snrs = [], [], []
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            waveform, target = self.prepare(batch)
            probabilities.append(predict_proba(waveform).float().cpu())
            labels.append(target.cpu())
            snrs.append(batch["snr_db"].cpu())
        return snr_scores(
            torch.cat(probabilities).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), self.num_classes
        )


class AudioSet36Noise(AudioSet40Dataset):
    """Clip nhiễu gốc AudioSet 36L: nhãn là chỉ số trong `class_names` (thứ tự labels.txt của Mixture-36l), cắt ngẫu nhiên khi train."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        root: str | Path | None = None,
        split: str = "train",
        duration: float = CLIP_SECONDS,
        exclude_ytids: set[str] | None = None,
        single_label: bool = True,
        random_crop: bool = True,
        dropped: Sequence[str] | None = None,
    ) -> None:
        super().__init__(root=root, split=split, duration=duration, exclude_ytids=exclude_ytids)
        dropped = dropped_label_names(class_names) if dropped is None else list(dropped)
        removed = filter_noise_36l(self, dropped)
        index40 = [self.class_names.index(name) for name in class_names]
        samples, multi = [], 0
        for path, target in self.samples:
            target36 = target[index40]
            hot = int(target36.sum())
            if hot == 0:
                continue
            if hot > 1:
                multi += 1
                if single_label:
                    continue
                target36 = torch.nn.functional.one_hot(target36.argmax(), len(index40)).float()
            samples.append((path, target36))
        self.samples = samples
        self.class_names = list(class_names)
        self.random_crop = random_crop
        logger.info(
            "%s: AudioSet 36L: bỏ %d clip mang nhãn %s, %d clip nhiều nhãn %s, còn %d clip",
            split, removed, dropped, multi, "bị loại" if single_label else "lấy nhãn đầu", len(samples),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, target = self.samples[index]
        rng = np.random.default_rng() if self.random_crop else None
        waveform = self.read(path, rng)
        return {"waveform": waveform, "noise": waveform, "target": target, "snr_db": torch.tensor(0.0)}


def build_loaders(config: BEATTrainConfig, max_samples: int | None, workers: int, single_label: bool = True) -> dict[str, Any]:
    class_names = mixture_class_names()
    held_out = read_noise_ytids(splits=("validation", "test"))
    train = AudioSet36Noise(class_names, split="train", exclude_ytids=held_out, single_label=single_label)
    validation = Mixture36Dataset(split="validation", duration=CLIP_SECONDS, load_noise=True)
    if train.class_names != validation.class_names:
        raise ValueError("thứ tự lớp của AudioSet 36L khác labels.txt của Mixture-36l")
    logger.info("train %d clip AudioSet 36L (đã loại %d ytid validation/test) | validation %d clip noise_path", len(train), len(held_out), len(validation))
    return {
        "train": build_dataloader(subsample(train, max_samples), config.batch_size, True, workers),
        "validation": build_dataloader(subsample(validation, max_samples), config.eval_batch_size, False, workers),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviewkd-config", type=Path, default=ROOT / "config" / "reviewkd.yaml", help="lấy target_rms và đường dẫn BEATs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 64 mẫu")
    parser.add_argument("--keep-multilabel", action="store_true", help="giữ clip nhiều nhãn (lấy nhãn có chỉ số nhỏ nhất)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-drive", action="store_true")
    parser.add_argument("--drive-resume-every", type=int, default=5, help="upload resume.pt (1,4 GB) mỗi N epoch; best.pt vẫn upload mỗi lần cải thiện")
    parser.add_argument("--keep-history", action="store_true", help="giữ thêm bản best_e<epoch>_<UTC>.pt trên Drive (mỗi bản ~360 MB)")
    parser.add_argument("--drive-dir", default=DEFAULT_DRIVE_DIR)
    parser.add_argument("--download-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "output" / "beats-noise-ft")
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.work_dir)
    raw = read_reviewkd_config(args.reviewkd_config)
    target_rms = float(raw["review"]["target_rms"])
    if target_rms <= 0:
        raise ValueError("review.target_rms must be positive for the noise teacher")
    logger.info("mixture-36l: %s", ensure_dataset(MIXTURE_KEY, args.download_root))
    logger.info("audioset-40l: %s", ensure_dataset(AUDIOSET_KEY, args.download_root))
    beats_path = resolve_path(raw["teacher"]["beats_checkpoint"])
    if not beats_path.is_file():
        beats_path = download_checkpoint(resolve_url("beats-pretrain"), beats_path.parent)

    config = BEATTrainConfig.from_yaml(BEAT_CONFIG)
    overrides = {k: v for k, v in (("epochs", args.epochs), ("patience", args.patience), ("lr", args.lr)) if v is not None}
    config = replace(config, **overrides)
    max_samples, workers = args.max_samples, args.workers
    if args.smoke:
        config = replace(config, epochs=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
        max_samples, workers = SMOKE_SAMPLES, 0
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = None
    if args.use_drive and not args.smoke:
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, token="cache")
    stage = STAGE_NAME + ("_smoke" if args.smoke else "")
    files = prepare_stage(stage, args.work_dir / "checkpoint", logger, fs, args.drive_dir, keep_history=args.keep_history)
    if files.drive_backup is not None and args.drive_resume_every > 1:
        files.drive_backup = every_n_calls(files.drive_backup, args.drive_resume_every)
    if files.result.is_file():
        logger.info("%s: finished earlier, reading %s", stage, files.result)
        return json.loads(files.result.read_text(encoding="utf-8"))

    beats = BEATClassifier.from_yaml(BEAT_CONFIG)
    beats.load_pretrained(beats_path)
    set_trainable(beats, True)
    beats.to(device)
    logger.info("BEATs %.1fM params | target_rms %.5f | epochs %d | lr %.1e | batch %d | device %s",
                beats.num_parameters() / 1e6, target_rms, config.epochs, config.lr, config.batch_size, device)

    loaders = build_loaders(config, max_samples, workers, single_label=not args.keep_multilabel)
    trainer = NoiseBEATTrainer(
        beats, config, device, logger, STAGE_NAME, files.best, None, files.resume, files.drive_backup, files.on_checkpoint,
        target_rms=target_rms,
    )
    if files.resume.is_file():
        trainer.load_resume(files.resume)
    fit = trainer.fit(loaders["train"], loaders["validation"])
    trainer._in_fit = False
    val = trainer.evaluate(loaders["validation"], desc="beat_noise - validation")
    logger.info("beat_noise: validation | " + " ".join(f"{name} {val[name]:.4f}" for name in METRIC_NAMES))

    result = {**fit.to_dict(), "history": trainer.history, "val": val, "target_rms": target_rms}
    save_result(files, result)
    return result


if __name__ == "__main__":
    main()
