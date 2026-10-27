import matplotlib
matplotlib.use("Agg")      # chay nen: khong co man hinh

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from tqdm.auto import tqdm

ROOT = Path.cwd().resolve()
if not (ROOT / "scripts").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_dataset import ensure_dataset
from scripts.finetune import (BEAT_CONFIG, CLIP_SECONDS, METRIC_NAMES, SMOKE_BATCH, SMOKE_SAMPLES,
                              prepare_stage, save_result, set_trainable)
from scripts.load_dataset import MIXTURE_KEY, Mixture36Dataset, build_dataloader, subsample
from scripts.load_pretrain import download_checkpoint, resolve_url
from scripts.drive_sync import DriveFileBackup
from scripts.run_beats_noise import every_n_calls
from scripts.run_frcrn_noise import setup_logging
from scripts.run_kd_beat import SNR_LEVELS, format_table, per_snr_table, write_per_snr_csv
from scripts.run_reviewkd import read_reviewkd_config, resolve_path
from src.metrics import ClassificationMetrics
from src.model.beat import BEATClassifier
from src.model.frcrn import FRCRNSeparator
from src.model.reviewkd import rms_normalize
from src.train import BEATTrainConfig, BEATTrainer
from src.train.beat_trainer import cosine_with_warmup
from src.train.reviewkd_trainer import snr_scores
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed

# ---- Cờ chạy: mặc định TẮT để không vô tình huấn luyện/nạp nặng trên máy local ----
RUN_FINETUNE = False         # bước 0: finetune BEATs trên n̂ của FRCRN
SMOKE = False                 # True: 1 epoch, 64 mẫu (checkpoint ghi vào stage "<FINETUNE_STAGE>_smoke")
LOAD_REAL = True             # bước 1: dựng student + teacher thật và kiểm tra trên CHECK_BATCHES batch
RUN_TRAIN = True             # bước 2: huấn luyện student, đánh giá 9 metrics theo SNR mỗi epoch, vẽ đường cong
BIO_VARIANT = "edge"          # cỡ BioME: "edge" (~6M, dim 192) hoặc "small" (~26M, dim 384); xem BIO_VARIANTS ở cell BioConfig
TRAIN_STUDENT_FRONT = False   # True: FRCRN của student huấn luyện được (khi đó teacher và student dùng hai bản riêng)
CHECK_BATCHES = 4             # số batch validation dùng ở bước kiểm tra
USE_DRIVE = False            # True: khôi phục và backup best.pt/resume.pt/history lên Google Drive (đăng nhập Drive một lần trong notebook)
DRIVE_DIR = "FA26_Capstone/Truong/beats-frcrn-ft"   # BEATs: <DRIVE_DIR>/<FINETUNE_STAGE>/; student: <DRIVE_DIR>/biome-<BIO_VARIANT>/
FT_EPOCHS = 60              # số epoch finetune BEATs; None: lấy từ config/beat.yaml (molab đặt 60)
FT_PATIENCE = 10            # patience finetune BEATs; None: lấy từ config/beat.yaml (molab đặt 10)
RESUME_EVERY = 5              # upload resume.pt (~1,4 GB) lên Drive mỗi N epoch; best.pt vẫn upload mỗi lần cải thiện

# ---- Gói chống overfit (so với lần chạy gốc: không augmentation, weight decay 0.01, patience 15, 100 epoch) ----
AUG_TAG = "aug"               # hậu tố thư mục kết quả: output/biome-kd/<BIO_VARIANT>-<AUG_TAG>; "" = đường dẫn gốc
EPOCHS_OVERRIDE = 8           # None: lấy từ config/reviewkd.yaml
PATIENCE_OVERRIDE = 5         # None: lấy từ config/reviewkd.yaml
WEIGHT_DECAY = 0.05           # lần gốc: 0.01; baseline CNN10 dùng 0.05
CROP_SECONDS = 3.0            # cắt ngẫu nhiên khi train (teacher và student thấy cùng đoạn); None = giữ nguyên 4 s
SPEC_AUG = True               # SpecAugment trên fbank của student: 2 mặt nạ tần số (<=16 bin) và 2 mặt nạ thời gian (<=24 khung)
DROP_PATH = 0.1               # stochastic depth cực đại, tăng tuyến tính theo layer
HEAD_DROPOUT = 0.1            # dropout trước đầu phân loại

# ---- Đường dẫn và hằng số dùng chung (một nơi duy nhất) ----
FRCRN_E37 = ROOT / "pretrained" / "FRCRN" / "best_e037_20261020T203958Z.pt"   # FRCRN epoch 37
WORK_DIR = ROOT / "output" / "beats-frcrn-ft"
FINETUNE_STAGE = "beat_frcrn" + ("_smoke" if SMOKE else "")
BEATS_FT = WORK_DIR / "checkpoint" / FINETUNE_STAGE / "best.pt"
BIO_DIR = ROOT / "output" / "biome-kd" / (BIO_VARIANT + (f"-{AUG_TAG}" if AUG_TAG else ""))                                     # kết quả student: best.pt, history, bảng, hình
HISTORY_PATH = BIO_DIR / ("history_smoke.json" if SMOKE else "history.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_rms = float(read_reviewkd_config(ROOT / "config" / "reviewkd.yaml")["review"]["target_rms"])
torch.set_float32_matmul_precision("high")
WORKERS = min(8, os.cpu_count() or 1)
print(f"ROOT {ROOT} | device {device} | target_rms {target_rms:.5f} | FRCRN {FRCRN_E37.name} | BEATs ft {BEATS_FT}")


def open_drive():
    # Kết nối Drive bằng token đã lưu; None khi tắt USE_DRIVE hoặc đang SMOKE.
    if not USE_DRIVE or SMOKE:
        return None
    from gdrive_fsspec import GoogleDriveFileSystem

    return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, token="cache")


@contextmanager
def stage(name: str):
    # In mốc bắt đầu/kết thúc kèm thời gian để thấy notebook đang ở bước nào.
    start = time.perf_counter()
    print(f"[..] {name}")
    yield
    print(f"[ok] {name} ({time.perf_counter() - start:.1f}s)")


class FRCRNBEATTrainer(BEATTrainer):
    # Finetune BEATs trên n̂ = mixture − ŝ do FRCRN đóng băng tạo ra, chuẩn hoá RMS như lúc suy luận/KD.
    def __init__(self, *args: Any, separator: torch.nn.Module, target_rms: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.separator, self.target_rms = separator, target_rms

    def prepare(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        mixture = batch["waveform"].to(self.device)
        with torch.no_grad():
            noise_hat = self.separator(mixture)
        return rms_normalize(noise_hat, self.target_rms), batch["target"].argmax(dim=1).to(self.device)
if RUN_FINETUNE:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(WORK_DIR)                                   # handler stdout + train.log: tóm tắt từng epoch hiện trong notebook
    logger = logging.getLogger("beats_frcrn_ft")
    set_seed(42)

    config = BEATTrainConfig.from_yaml(BEAT_CONFIG)
    _max_samples, _workers = None, WORKERS
    if SMOKE:
        config = replace(config, epochs=1, batch_size=SMOKE_BATCH, eval_batch_size=SMOKE_BATCH)
        _max_samples, _workers = SMOKE_SAMPLES, 0
    overrides = {k: v for k, v in (("epochs", FT_EPOCHS), ("patience", FT_PATIENCE)) if v is not None}
    config = replace(config, **overrides)
    print(f"finetune | stage {FINETUNE_STAGE} | epochs {config.epochs} | batch {config.batch_size} | lr {config.lr:.1e} | smoke {SMOKE}")

    with tqdm(total=5, desc="finetune: dựng", unit="bước") as _bar:
        with stage("dữ liệu Mixture-36l"):
            ensure_dataset(MIXTURE_KEY, ROOT / "datasets")
            _loaders = {
                "train": build_dataloader(subsample(Mixture36Dataset(split="train", duration=CLIP_SECONDS), _max_samples), config.batch_size, True, _workers),
                "validation": build_dataloader(subsample(Mixture36Dataset(split="validation", duration=CLIP_SECONDS), _max_samples), config.eval_batch_size, False, _workers),
            }
            print(f"  train {len(_loaders['train'].dataset)} clip | validation {len(_loaders['validation'].dataset)} clip")
        _bar.update()

        with stage("FRCRN epoch 37 (đóng băng)"):
            separator = FRCRNSeparator.from_yaml(ROOT / "config" / "frcrn.yaml")
            separator.load_pretrained(FRCRN_E37)
            separator.to(device).freeze()
            print(f"  FRCRN {separator.num_parameters() / 1e6:.1f}M tham số")
        _bar.update()

        with stage("BEATs pretrain"):
            raw = read_reviewkd_config(ROOT / "config" / "reviewkd.yaml")
            beats_path = resolve_path(raw["teacher"]["beats_checkpoint"])
            if not beats_path.is_file():
                beats_path = download_checkpoint(resolve_url("beats-pretrain"), beats_path.parent)
            beats = BEATClassifier.from_yaml(BEAT_CONFIG)
            beats.load_pretrained(beats_path)
            set_trainable(beats, True)
            beats.to(device)
            print(f"  BEATs {beats.num_parameters() / 1e6:.1f}M tham số, huấn luyện {beats.num_parameters(trainable_only=True) / 1e6:.1f}M")
        _bar.update()

        with stage("checkpoint/resume"):
            _fs = open_drive()
            files = prepare_stage(FINETUNE_STAGE, WORK_DIR / "checkpoint", logger, _fs, DRIVE_DIR, keep_history=False)
            if files.drive_backup is not None and RESUME_EVERY > 1:
                files.drive_backup = every_n_calls(files.drive_backup, RESUME_EVERY)
            print(f"  Drive {'bật: ' + DRIVE_DIR if _fs is not None else 'tắt'} | best {'có' if files.best.is_file() else 'chưa có'} | resume {'có' if files.resume.is_file() else 'chưa có'} | xong trước đó {'có' if files.result.is_file() else 'chưa'}")
        _bar.update()

        with stage("trainer"):
            trainer = FRCRNBEATTrainer(
                beats, config, device, logger, FINETUNE_STAGE, files.best, None, files.resume, files.drive_backup, files.on_checkpoint,
                separator=separator, target_rms=target_rms,
            )
            if files.resume.is_file():
                trainer.load_resume(files.resume)
        _bar.update()

    if files.result.is_file():
        print(f"finetune đã xong trước đó, bỏ qua huấn luyện: {files.result}")
    else:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _t0 = time.perf_counter()
        fit = trainer.fit(_loaders["train"], _loaders["validation"])       # tqdm theo batch nằm trong trainer
        trainer._in_fit = False
        print(f"fit xong sau {time.perf_counter() - _t0:.0f}s | {fit.to_dict()}")
        if device.type == "cuda":
            print(f"VRAM đỉnh: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        val = trainer.evaluate(_loaders["validation"], desc="beat_frcrn - validation")
        print("validation | " + " ".join(f"{name} {val[name]:.4f}" for name in METRIC_NAMES))
        save_result(files, {**fit.to_dict(), "history": trainer.history, "val": val, "target_rms": target_rms})   # ghi result.json + backup Drive
        print("best checkpoint:", files.best, "| đã backup Drive" if _fs is not None else "")
else:
    print("RUN_FINETUNE = False: bỏ qua finetune (đặt True trên GPU để chạy).")
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import torch.nn.functional as F
from torch import nn      # torch và Tensor đã import ở cell cấu hình


@dataclass(frozen=True)
class BioConfig:
    dim: int = 192              # chiều ẩn: Edge 192 (bài báo), Small 384, Base 768 (xem BIO_VARIANTS)
    layers: int = 12            # bằng độ sâu của teacher BEATs
    heads: int = 3              # head_dim = 64
    kv_heads: int = 1           # GQA: 3 query head chia sẻ 1 nhóm key/value
    ffn_hidden: int = 512
    n_mels: int = 128
    patch: int = 16
    mod_bands: int = 64         # số dải trung bình của trục modulation trong MSAB
    teacher_dim: int = 768
    kd_layers: tuple = (3, 6, 9, 12)
    num_classes: int = 36
    sample_rate: int = 16000
    drop_path: float = 0.0       # stochastic depth cực đại (layer cuối); 0 = tắt
    head_dropout: float = 0.0
    spec_aug: bool = False

    @property
    def side_dim(self) -> int:
        return self.n_mels + self.mod_bands


# Cỡ BioME. Bài báo (arXiv 2602.09970) nêu Edge 6M/dim 192, Small 26M, Base 76M/dim 768 và chiều sâu 12 như teacher;
# số head, nhóm KV và FFN KHÔNG được nêu: các giá trị dưới là lựa chọn của người viết, chọn để tổng tham số gần mức bài báo.
BIO_VARIANTS = {
    "edge": dict(dim=192, heads=3, kv_heads=1, ffn_hidden=512),     # ước lượng tay ~6,3M
    "small": dict(dim=384, heads=6, kv_heads=2, ffn_hidden=1024),   # ước lượng tay ~22M (bài: 26M)
}

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


@lru_cache(maxsize=8)
def rope_cache(length: int, head_dim: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(length, device=device).float(), inv_freq)   # (L, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, L, head_dim); xoay từng cặp chiều (2i, 2i+1) theo góc phụ thuộc vị trí
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class GQA(nn.Module):
    def __init__(self, cfg: BioConfig):
        super().__init__()
        self.h, self.g, self.hd = cfg.heads, cfg.kv_heads, cfg.dim // cfg.heads
        self.q = nn.Linear(cfg.dim, self.h * self.hd, bias=False)
        self.k = nn.Linear(cfg.dim, self.g * self.hd, bias=False)
        self.v = nn.Linear(cfg.dim, self.g * self.hd, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, length, _ = x.shape
        q = self.q(x).view(b, length, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(b, length, self.g, self.hd).transpose(1, 2)
        v = self.v(x).view(b, length, self.g, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        try:        # torch >= 2.5: SDPA tự chia sẻ KV theo nhóm, không sao chép K/V ra h/g lần
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        except TypeError:
            k, v = k.repeat_interleave(self.h // self.g, dim=1), v.repeat_interleave(self.h // self.g, dim=1)
            out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(b, length, -1))


class SwiGLU(nn.Module):
    # FFN kiểu Llama: W3( SiLU(W1 x) * (W2 x) )
    def __init__(self, cfg: BioConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class BioLayer(nn.Module):
    def __init__(self, cfg: BioConfig, drop_path: float = 0.0):
        super().__init__()
        self.drop_path = drop_path
        self.norm_attn, self.norm_film, self.norm_ffn = RMSNorm(cfg.dim), RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attn, self.ffn = GQA(cfg), SwiGLU(cfg)
        self.film = nn.Linear(cfg.side_dim, 2 * cfg.dim)     # (gamma, beta) = Linear(h)
        nn.init.zeros_(self.film.weight)                      # khởi tạo khối FiLM thành no-op để huấn luyện ổn định
        nn.init.zeros_(self.film.bias)

    def _branch(self, y: Tensor) -> Tensor:
        # stochastic depth theo mẫu: bỏ nhánh với xác suất drop_path khi huấn luyện, chia lại cho xác suất giữ
        if not self.training or self.drop_path == 0.0:
            return y
        keep = 1.0 - self.drop_path
        return y * ((torch.rand(y.shape[0], 1, 1, device=y.device) < keep).to(y.dtype) / keep)

    def forward(self, x: Tensor, side: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self._branch(self.attn(self.norm_attn(x), cos, sin))
        gamma, beta = self.film(side).unsqueeze(1).chunk(2, dim=-1)
        x = x + gamma * self.norm_film(x) + beta              # x' = gamma ⊙ x + beta, rồi cộng residual
        return x + self._branch(self.ffn(self.norm_ffn(x)))
class BioFrontend(nn.Module):
    # n̂ (B, N) -> fbank (B, T, n_mels) cho patch embedding  và  MSAB h (B, side_dim)  cho FiLM. Toàn bộ khả vi.
    def __init__(self, cfg: BioConfig, n_fft: int = 400, hop: int = 160, fmin: float = 20.0,
                 mean: float = 15.41663, std: float = 6.55582):
        super().__init__()
        import torchaudio.functional as taf

        self.cfg, self.n_fft, self.hop, self.mean, self.std = cfg, n_fft, hop, mean, std
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("fb", taf.melscale_fbanks(n_fft // 2 + 1, fmin, cfg.sample_rate / 2, cfg.n_mels,
                                                       cfg.sample_rate, norm=None, mel_scale="htk"))
        self.side_norm = nn.LayerNorm(cfg.side_dim)

    def forward(self, wave: Tensor):
        with torch.autocast(device_type=wave.device.type, enabled=False):      # STFT/FFT n_fft=400 luôn chạy fp32
            return self._forward(wave.float())

    def _forward(self, wave: Tensor):
        x = wave * 32768.0
        x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)                     # pre-emphasis như fbank của BEATs
        spec = torch.stft(x, self.n_fft, self.hop, window=self.window, center=False, return_complex=True)
        mel = (spec.real.square() + spec.imag.square()).transpose(1, 2) @ self.fb          # (B, T, n_mels)
        fbank = (torch.log(mel.clamp_min(1e-7)) - self.mean) / (2 * self.std)

        # MSAB: phổ modulation = FFT theo thời gian của đường bao mel; lấy trung bình theo từng trục rồi nối.
        env = mel.sqrt()
        modulation = torch.fft.rfft(env - env.mean(dim=1, keepdim=True), dim=1).abs()      # (B, T//2+1, n_mels)
        per_mel = modulation.mean(dim=1)                                                    # (B, n_mels)
        per_mod = F.adaptive_avg_pool1d(modulation.mean(dim=2).unsqueeze(1), self.cfg.mod_bands).squeeze(1)
        return fbank, self.side_norm(torch.log1p(torch.cat([per_mel, per_mod], dim=-1)))


class MaskedFrontEnd(nn.Module):
    # Giao diện FRCRN dạng masking. `mask_net(real, imag) -> (mask_re, mask_im)` là FRCRN thật (src/model/frcrn);
    # bản thật còn chuẩn hoá mức âm lượng (audio_norm) quanh bước này.
    def __init__(self, mask_net: nn.Module, n_fft: int = 640, hop: int = 320):
        super().__init__()
        self.mask_net, self.n_fft, self.hop = mask_net, n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft).sqrt())

    def forward(self, mixture: Tensor):
        spec = torch.stft(mixture, self.n_fft, self.hop, window=self.window, return_complex=True)   # (B, F, T)
        mask_re, mask_im = self.mask_net(spec.real, spec.imag)
        speech = torch.complex(spec.real * mask_re - spec.imag * mask_im, spec.real * mask_im + spec.imag * mask_re)
        s_hat = torch.istft(speech, self.n_fft, self.hop, window=self.window, length=mixture.shape[-1])
        return mixture - s_hat, s_hat                                                         # (n̂, ŝ)


class HalfMask(nn.Module):
    # Mặt nạ giả chỉ để kiểm tra shape và gradient; thay bằng FRCRN thật.
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, real: Tensor, imag: Tensor):
        return self.scale * torch.ones_like(real), torch.zeros_like(imag)
def spec_augment(fbank: Tensor, freq_masks: int = 2, freq_width: int = 16, time_masks: int = 2, time_width: int = 24) -> Tensor:
    # fbank (B, T, F) đã chuẩn hoá: đặt 0 (= mức trung bình) các dải tần số và khoảng thời gian ngẫu nhiên, riêng từng mẫu.
    b, t, f = fbank.shape
    out = fbank
    freq_idx = torch.arange(f, device=fbank.device)[None]
    time_idx = torch.arange(t, device=fbank.device)[None]
    for _ in range(freq_masks):
        width = torch.randint(0, freq_width + 1, (b, 1), device=fbank.device)
        start = (torch.rand(b, 1, device=fbank.device) * (f - width)).long()
        out = out.masked_fill(((freq_idx >= start) & (freq_idx < start + width))[:, None, :], 0.0)
    for _ in range(time_masks):
        width = torch.randint(0, time_width + 1, (b, 1), device=fbank.device)
        start = (torch.rand(b, 1, device=fbank.device) * (t - width)).long()
        out = out.masked_fill(((time_idx >= start) & (time_idx < start + width))[:, :, None], 0.0)
    return out


class StudentOutput(NamedTuple):
    logits: Tensor
    projected: dict      # {k: (B, L, teacher_dim)} cho k trong cfg.kd_layers
    noise_hat: Tensor
    side: Tensor


class BioStudent(nn.Module):
    def __init__(self, front: nn.Module, cfg: BioConfig = BioConfig()):
        super().__init__()
        self.front, self.cfg = front, cfg
        self.frontend = BioFrontend(cfg)
        self.patch = nn.Conv2d(1, cfg.dim, cfg.patch, stride=cfg.patch)          # = chiếu tuyến tính từng patch 16×16
        self.layers = nn.ModuleList(BioLayer(cfg, cfg.drop_path * i / max(cfg.layers - 1, 1)) for i in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim)
        self.head_drop = nn.Dropout(cfg.head_dropout)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)
        self.proj = nn.ModuleDict({str(k): nn.Linear(cfg.dim, cfg.teacher_dim) for k in cfg.kd_layers})

    def forward(self, mixture: Tensor, noise_hat: Tensor | None = None) -> StudentOutput:
        if noise_hat is None:                  # truyền n̂ vào để khỏi chạy lại FRCRN khi đã tính ở nơi khác
            noise_hat, _ = self.front(mixture)
        fbank, side = self.frontend(noise_hat)
        if self.training and self.cfg.spec_aug:
            fbank = spec_augment(fbank)
        x = self.patch(fbank.unsqueeze(1)).flatten(2).transpose(1, 2)           # (B, Tp*8, dim), token theo thời gian như BEATs
        cos, sin = rope_cache(x.shape[1], self.cfg.dim // self.cfg.heads, x.device)
        projected = {}
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x, side, cos, sin)
            if i in self.cfg.kd_layers:
                projected[i] = self.proj[str(i)](x)
        return StudentOutput(self.head(self.head_drop(self.norm(x).mean(dim=1))), projected, noise_hat, side)
class TeacherOutput(NamedTuple):
    # Teacher = mixture -> FRCRN (đóng băng) -> BEATs đã finetune; chạy dưới no_grad.
    logits: Tensor       # (B, 36)
    hidden: dict         # {k: (B, L, teacher_dim)} với k trong cfg.kd_layers


def layerwise_kd_loss(projected: dict, teacher_hidden: dict):
    # L = mean_k mean_t [ mean_d |ẑ - z| - log sigmoid(cos(ẑ, z)) ]   (công thức chưng cất của BioME)
    terms = {}
    for k, z_hat in projected.items():
        z = teacher_hidden[k].detach().float()
        z_hat = z_hat.float()
        l1 = (z_hat - z).abs().mean(dim=-1)
        terms[k] = (l1 - F.logsigmoid(F.cosine_similarity(z_hat, z, dim=-1))).mean()
    return torch.stack(list(terms.values())).mean(), terms


def total_loss(out: StudentOutput, teacher: TeacherOutput, labels: Tensor,
               w_ce: float = 1.0, w_kd: float = 1.0, w_feat: float = 1.0, temperature: float = 2.0):
    # BioME gốc chỉ chưng cất đặc trưng (không nhãn); ở đây thêm CE và KD logit vì bài toán có nhãn.
    logits = out.logits.float()                  # loss luôn tính fp32 dù encoder chạy bf16
    ce = F.cross_entropy(logits, labels, label_smoothing=0.1)
    kd = F.kl_div(F.log_softmax(logits / temperature, -1), F.softmax(teacher.logits.float() / temperature, -1),
                  reduction="batchmean") * temperature**2
    feat, per_layer = layerwise_kd_loss(out.projected, teacher.hidden)
    parts = {"ce": ce, "kd": kd, "feat": feat, **{f"feat_layer{k}": v for k, v in per_layer.items()}}
    return w_ce * ce + w_kd * kd + w_feat * feat, parts
# Kiểm tra shape trên dữ liệu giả (chưa chạy trong phiên viết notebook này).
_cfg = BioConfig(**BIO_VARIANTS[BIO_VARIANT], drop_path=DROP_PATH, head_dropout=HEAD_DROPOUT, spec_aug=SPEC_AUG)
print(f"BioME {BIO_VARIANT}: dim {_cfg.dim}, {_cfg.layers} layer, {_cfg.heads} head / {_cfg.kv_heads} KV, FFN {_cfg.ffn_hidden}")
_student = BioStudent(MaskedFrontEnd(HalfMask()), _cfg)
_mixture = torch.randn(2, 64000) * 0.1                                  # 4 giây, 16 kHz
_out = _student(_mixture)

_frames = (64000 - 400) // 160 + 1                                      # 398 khung
_tokens = (_frames // _cfg.patch) * (_cfg.n_mels // _cfg.patch)             # 24 × 8 = 192, trùng lưới token của BEATs
assert _out.logits.shape == (2, _cfg.num_classes)
assert all(_out.projected[k].shape == (2, _tokens, _cfg.teacher_dim) for k in _cfg.kd_layers)

_teacher = TeacherOutput(torch.randn(2, _cfg.num_classes), {k: torch.randn(2, _tokens, _cfg.teacher_dim) for k in _cfg.kd_layers})
_loss, _parts = total_loss(_out, _teacher, torch.tensor([0, 1]))
_loss.backward()

assert _student.front.mask_net.scale.grad is not None                    # gradient chảy ngược về FRCRN
print({k: round(float(v), 3) for k, v in _parts.items()})
print("tham số encoder (không gồm FRCRN):", round(sum(p.numel() for n, p in _student.named_parameters() if not n.startswith("front.")) / 1e6, 2), "M")
class FRCRNFront(nn.Module):
    # Giao diện front(mixture) -> (n̂, ŝ) cho BioStudent. FRCRNSeparator (target=speech) trả thẳng n̂; ŝ = mixture − n̂.
    def __init__(self, separator: nn.Module, target_rms: float, frozen: bool = True):
        super().__init__()
        self.separator, self.target_rms, self.frozen = separator, target_rms, frozen

    def train(self, mode: bool = True):
        return super().train(False if self.frozen else mode)       # FRCRN có BatchNorm: bản đóng băng luôn ở eval

    def forward(self, mixture: Tensor):
        noise_hat = self.separator(mixture)
        return rms_normalize(noise_hat, self.target_rms), mixture - noise_hat


def load_frcrn(path: Path, device, freeze: bool) -> FRCRNSeparator:
    separator = FRCRNSeparator.from_yaml(ROOT / "config" / "frcrn.yaml")
    separator.load_pretrained(path)
    return separator.to(device).freeze() if freeze else separator.to(device)


class BioTeacher(nn.Module):
    # n̂ -> BEATs đã finetune (đóng băng). Chạy dưới no_grad (không dùng inference_mode: loss còn lưu tensor của teacher cho backward).
    def __init__(self, front: FRCRNFront, beats: BEATClassifier, cfg: BioConfig):
        super().__init__()
        self.front, self.beats, self.cfg = front, beats, cfg

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, mixture: Tensor, noise_hat: Tensor | None = None) -> TeacherOutput:
        if noise_hat is None:
            noise_hat, _ = self.front(mixture)
        out = self.beats(noise_hat, layers=self.cfg.kd_layers)
        return TeacherOutput(out.logits, dict(zip(self.cfg.kd_layers, out.hidden)))


def build_models(cfg: BioConfig):
    # Trả (student, teacher, shared). shared=True: hai bên dùng chung một FRCRN đóng băng nên chỉ cần chạy nó một lần.
    shared = not TRAIN_STUDENT_FRONT
    with tqdm(total=3, desc="dựng student/teacher", unit="bước") as bar:
        with stage("FRCRN epoch 37"):
            teacher_front = FRCRNFront(load_frcrn(FRCRN_E37, device, freeze=True), target_rms)
            student_front = teacher_front if shared else FRCRNFront(load_frcrn(FRCRN_E37, device, freeze=False), target_rms, frozen=False)
            print(f"  FRCRN {'dùng chung' if shared else 'hai bản (student huấn luyện được)'}")
        bar.update()
        with stage(f"BEATs đã finetune {BEATS_FT.name}"):
            if not BEATS_FT.is_file():
                raise FileNotFoundError(f"chưa có {BEATS_FT}: chạy bước 0 (RUN_FINETUNE) hoặc chỉnh BEATS_FT")
            beats = BEATClassifier.from_yaml(BEAT_CONFIG)
            load_checkpoint(BEATS_FT, beats)
            beats.to(device).freeze()
            if SMOKE:
                print("  CẢNH BÁO: SMOKE = True, đây là checkpoint smoke (1 epoch, 64 mẫu), chưa phải teacher thật")
        bar.update()
        with stage("student BioME"):
            student = BioStudent(student_front, cfg).to(device)
            encoder = sum(p.numel() for n, p in student.named_parameters() if not n.startswith("front.")) / 1e6
            print(f"  tham số encoder (không gồm FRCRN): {encoder:.2f}M")
        bar.update()
    return student, BioTeacher(teacher_front, beats, cfg), shared


def random_crop(wave: Tensor, length: int) -> Tensor:
    # Cắt ngẫu nhiên `length` mẫu cho từng clip trong batch; teacher và student nhận cùng đoạn vì cắt trước khi vào cả hai.
    if wave.shape[1] <= length:
        return wave
    start = torch.randint(0, wave.shape[1] - length + 1, (wave.shape[0],), device=wave.device)
    return wave.gather(1, start[:, None] + torch.arange(length, device=wave.device)[None])


def distill_forward(student: BioStudent, teacher: BioTeacher, mixture: Tensor, shared: bool):
    # Một bước forward của student và teacher; FRCRN chỉ chạy một lần khi dùng chung. Encoder student chạy bf16 trên GPU.
    noise_hat = None
    if shared:
        with torch.no_grad():
            noise_hat, _ = student.front(mixture)
    teacher_out = teacher(mixture, noise_hat)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        student_out = student(mixture, noise_hat)
    return student_out, teacher_out
MODELS = None
if LOAD_REAL or RUN_TRAIN:
    cfg = BioConfig(**BIO_VARIANTS[BIO_VARIANT], drop_path=DROP_PATH, head_dropout=HEAD_DROPOUT, spec_aug=SPEC_AUG)
    MODELS = build_models(cfg)

if LOAD_REAL:
    _student, _teacher, _shared = MODELS
    _student.train()

    _loader = build_dataloader(subsample(Mixture36Dataset(split="validation", duration=CLIP_SECONDS), CHECK_BATCHES * 8), 8, False, 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _correct = _seen = 0
    _sums: dict[str, float] = {}
    _bar = tqdm(_loader, desc="kiểm tra student/teacher", unit="batch", total=CHECK_BATCHES)
    for _step, _batch in enumerate(_bar, start=1):
        _mixture, _labels = _batch["waveform"].to(device), _batch["target"].argmax(dim=1).to(device)
        _student_out, _teacher_out = distill_forward(_student, _teacher, _mixture, _shared)
        _loss, _parts = total_loss(_student_out, _teacher_out, _labels)
        if _step == 1:                                                    # backward một lần: kiểm gradient chảy về encoder
            _loss.backward()
            _grad = torch.sqrt(sum(p.grad.square().sum() for p in _student.parameters() if p.grad is not None))
            print(f"shape ok: logits {tuple(_student_out.logits.shape)} | token {_student_out.projected[cfg.kd_layers[0]].shape[1]} | grad norm {float(_grad):.3f}")
            _student.zero_grad(set_to_none=True)
        _correct += int((_teacher_out.logits.argmax(-1) == _labels).sum())
        _seen += _labels.numel()
        for name, value in _parts.items():
            _sums[name] = _sums.get(name, 0.0) + float(value)
        _bar.set_postfix(teacher_top1=f"{_correct / _seen:.3f}", loss=f"{float(_loss):.3f}", feat=f"{_sums['feat'] / _step:.3f}")
    print(f"teacher top-1 trên {_seen} clip validation: {_correct / _seen:.4f}")
    print("trung bình loss:", {k: round(v / _step, 3) for k, v in _sums.items()})
    if device.type == "cuda":
        print(f"VRAM đỉnh: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
else:
    print("LOAD_REAL = False: bỏ qua kiểm tra student/teacher thật (đặt True trên GPU để chạy).")
NUM_CLASSES = 36


@torch.no_grad()
def evaluate_scores(loader, logits_fn, desc: str) -> dict[str, float]:
    # Một lượt duyệt validation -> snr_scores: 9 chỉ số tổng + 9 chỉ số theo từng mức SNR (khoá <chỉ số>_snr_<mức>).
    probs, labels, snrs = [], [], []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
        mixture = batch["waveform"].to(device)
        probs.append(torch.softmax(logits_fn(mixture).float(), dim=-1).cpu())
        labels.append(batch["target"].argmax(dim=1))
        snrs.append(batch["snr_db"])
    scores = snr_scores(torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(snrs).numpy(), NUM_CLASSES)
    return {k: float(v) for k, v in scores.items()}


def student_logits(student: BioStudent):
    def run(mixture: Tensor) -> Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return student(mixture).logits
    return run


def epoch_line(entry: dict) -> str:
    val = entry["val"]
    per_snr = " ".join(f"{snr:+d}dB {val[f'acc_snr_{snr}']:.3f}" for snr in SNR_LEVELS)
    return (f"epoch {entry['epoch']:3d} | loss {entry['train_loss']:.4f} | train acc {entry['train_acc']:.3f} mAP {entry['train_mAP']:.3f} | "
            f"val acc {val['acc']:.3f} mAP {val['mAP']:.3f} macro_f1 {val['macro_f1']:.3f} | acc theo SNR: {per_snr}")


def train_student(student, teacher, shared, loaders, hp: dict, history: list[dict], backups: dict | None = None) -> dict:
    # Huấn luyện student; mỗi epoch đánh giá validation, ghi history.json, giữ best.pt theo macro_f1, dừng sớm theo patience.
    BIO_DIR.mkdir(parents=True, exist_ok=True)
    params = [p for p in student.parameters() if p.requires_grad]            # FRCRN đóng băng không vào optimizer
    optimizer = AdamW(params, lr=hp["lr"], betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    scheduler = cosine_with_warmup(optimizer, hp["epochs"] * len(loaders["train"]), hp["warmup_ratio"], 0.0)
    best_metric, best_epoch, stale = -1.0, 0, 0
    epochs_bar = tqdm(range(1, hp["epochs"] + 1), desc="epoch", unit="epoch")
    for epoch in epochs_bar:
        student.train()
        train_metrics = ClassificationMetrics(num_classes=NUM_CLASSES)
        loss_sum, count = 0.0, 0
        bar = tqdm(loaders["train"], desc=f"epoch {epoch}/{hp['epochs']} | train", unit="batch", leave=False, dynamic_ncols=True)
        for batch in bar:
            mixture, labels = batch["waveform"].to(device), batch["target"].argmax(dim=1).to(device)
            if CROP_SECONDS:
                mixture = random_crop(mixture, int(CROP_SECONDS * student.cfg.sample_rate))
            student_out, teacher_out = distill_forward(student, teacher, mixture, shared)
            loss, _ = total_loss(student_out, teacher_out, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            train_metrics.update(labels.cpu().numpy(), torch.softmax(student_out.logits.detach().float(), dim=-1).cpu().numpy())
            loss_sum, count = loss_sum + float(loss) * labels.numel(), count + labels.numel()
            bar.set_postfix(loss=f"{loss_sum / count:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        train_result = train_metrics.compute()

        student.eval()
        val = evaluate_scores(loaders["validation"], student_logits(student), f"epoch {epoch}/{hp['epochs']} | validation")
        entry = {"epoch": epoch, "train_loss": loss_sum / count, "train_acc": float(train_result.top1_accuracy),
                 "train_mAP": float(train_result.mAP), "lr": scheduler.get_last_lr()[0], "val": val}
        history.append(entry)
        if val["macro_f1"] > best_metric:
            best_metric, best_epoch, stale = val["macro_f1"], epoch, 0
            best_file = BIO_DIR / ("best_smoke.pt" if SMOKE else "best.pt")
            torch.save({"model": student.state_dict(), "epoch": epoch, "macro_f1": best_metric}, best_file)
            if backups:
                backups["best"](best_file)                                    # backup Drive mỗi lần cải thiện
        else:
            stale += 1
        HISTORY_PATH.write_text(json.dumps({"history": history, "best_epoch": best_epoch, "best_macro_f1": best_metric}), encoding="utf-8")
        if backups:
            backups["history"](HISTORY_PATH)
        epochs_bar.set_postfix(val_acc=f"{val['acc']:.3f}", val_mAP=f"{val['mAP']:.3f}", best=f"{best_metric:.3f}@{best_epoch}")
        tqdm.write(epoch_line(entry) + f" | best macro_f1 {best_metric:.4f} @ {best_epoch} | stale {stale}/{hp['patience']}")
        if stale >= hp["patience"]:
            tqdm.write(f"dừng sớm: {hp['patience']} epoch không cải thiện macro_f1")
            break
    return {"best_epoch": best_epoch, "best_macro_f1": best_metric}


def print_snr_report(title: str, scores: dict[str, float], csv_name: str) -> None:
    # In bảng 9 chỉ số x từng mức SNR (+ dòng mean) và ghi csv.
    table = per_snr_table(scores)
    write_per_snr_csv(table, BIO_DIR / csv_name)
    print(f"\n{title}: 9 chỉ số tổng\n  " + " ".join(f"{name} {scores[name]:.4f}" for name in METRIC_NAMES))
    print(format_table(table))
def plot_history(history: list[dict], teacher: dict | None, best_epoch: int, path: Path | None = None) -> None:
    # Hai ô: top-1 accuracy và mAP theo epoch. Một trục y mỗi ô; SNR là một dãy xanh một sắc độ (nhạt -> đậm = SNR thấp -> cao).
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#6e7781", "#e4e7eb"
    ramp = [plt.cm.Blues(0.38 + 0.57 * i / (len(SNR_LEVELS) - 1)) for i in range(len(SNR_LEVELS))]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor="white", constrained_layout=True)
    for ax, (title, key, train_key) in zip(axes, [("Top-1 accuracy theo epoch", "acc", "train_acc"), ("mAP theo epoch", "mAP", "train_mAP")]):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for snr, color in zip(SNR_LEVELS, ramp):
            series = [h["val"][f"{key}_snr_{snr}"] for h in history]
            ax.plot(epochs, series, color=color, linewidth=1.3, marker="o", markersize=3)
            ax.annotate(f"{snr:+d} dB", (epochs[-1], series[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=muted, va="center")
        ax.plot(epochs, [h[train_key] for h in history], color=muted, linewidth=1.6, linestyle="--", label="train")
        ax.plot(epochs, [h["val"][key] for h in history], color=ink, linewidth=2.6, marker="o", markersize=4, label="validation (tổng)")
        if teacher is not None:
            ax.axhline(teacher[key], color=ink, linewidth=1.2, linestyle=":", label=f"teacher ({teacher[key]:.3f})")
        ax.axvline(best_epoch, color=muted, linewidth=0.8, alpha=0.6)
        ax.set_title(title, loc="left", fontsize=12, color=ink)
        ax.set_xlabel("epoch", color=ink)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3 + 0.12 * (epochs[-1] - epochs[0] + 1))
        ax.tick_params(colors=ink)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"BioME student trên Mixture-36l (epoch tốt nhất = {best_epoch}; các đường xanh là validation theo mức SNR)", x=0.01, ha="left", fontsize=10, color=muted)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, facecolor="white")
        print("đã lưu", path)
    plt.show()


if RUN_TRAIN:
    student, teacher, shared = MODELS
    hp_raw = read_reviewkd_config(ROOT / "config" / "reviewkd.yaml")["train"]
    hp = {"lr": float(hp_raw["lr"]), "epochs": int(hp_raw["epochs"]), "patience": int(hp_raw["patience"]),
          "warmup_ratio": float(hp_raw["warmup_ratio"]), "batch": int(hp_raw["batch_size"]), "eval_batch": int(hp_raw["eval_batch_size"])}
    max_samples, workers = None, WORKERS
    if EPOCHS_OVERRIDE:
        hp["epochs"] = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE:
        hp["patience"] = PATIENCE_OVERRIDE
    if SMOKE:
        hp.update(epochs=1, batch=SMOKE_BATCH, eval_batch=SMOKE_BATCH)
        max_samples, workers = SMOKE_SAMPLES, 0
    set_seed(int(hp_raw["seed"]))
    print(f"huấn luyện student | lr {hp['lr']:.1e} | epochs {hp['epochs']} | batch {hp['batch']} | patience {hp['patience']} | smoke {SMOKE}")

    with stage("dữ liệu Mixture-36l"):
        loaders = {
            "train": build_dataloader(subsample(Mixture36Dataset(split="train", duration=CLIP_SECONDS), max_samples), hp["batch"], True, workers),
            "validation": build_dataloader(subsample(Mixture36Dataset(split="validation", duration=CLIP_SECONDS), max_samples), hp["eval_batch"], False, workers),
        }
        print(f"  train {len(loaders['train'].dataset)} clip | validation {len(loaders['validation'].dataset)} clip")
    with stage("đánh giá teacher (mốc so sánh)"):
        teacher_scores = evaluate_scores(loaders["validation"], lambda mixture: teacher(mixture).logits, "teacher | validation")
        print(f"  teacher acc {teacher_scores['acc']:.4f} mAP {teacher_scores['mAP']:.4f} macro_f1 {teacher_scores['macro_f1']:.4f}")

    best_path = BIO_DIR / ("best_smoke.pt" if SMOKE else "best.pt")
    _fs = open_drive()
    backups = None
    if _fs is not None:
        _remote = f"{DRIVE_DIR}/biome-{BIO_VARIANT}"
        backups = {"best": DriveFileBackup(_fs, _remote, best_path.name), "history": DriveFileBackup(_fs, _remote, HISTORY_PATH.name)}
        print(f"Drive bật: backup student vào {_remote}")

    history: list[dict] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    summary = train_student(student, teacher, shared, loaders, hp, history, backups)
    print(f"huấn luyện xong sau {time.perf_counter() - t0:.0f}s | {summary}")
    if device.type == "cuda":
        print(f"VRAM đỉnh: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    load_checkpoint(best_path, student)                                   # đánh giá lại đúng epoch tốt nhất
    student.eval()
    best_scores = evaluate_scores(loaders["validation"], student_logits(student), "student (best) | validation")
    print_snr_report(f"student (epoch {summary['best_epoch']})", best_scores, "student_per_snr.csv")
    print_snr_report("teacher", teacher_scores, "teacher_per_snr.csv")
    HISTORY_PATH.write_text(json.dumps({"history": history, "teacher": teacher_scores, "student_best": best_scores, **summary}), encoding="utf-8")
    plot_history(history, teacher_scores, summary["best_epoch"], BIO_DIR / "curves.png")
    if backups:                                                           # backup bản cuối: history đầy đủ, bảng SNR, hình
        backups["history"](HISTORY_PATH)
        for _name in ("student_per_snr.csv", "teacher_per_snr.csv", "curves.png"):
            DriveFileBackup(_fs, f"{DRIVE_DIR}/biome-{BIO_VARIANT}", _name)(BIO_DIR / _name)
elif HISTORY_PATH.is_file():
    saved = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))          # vẽ lại từ kết quả đã lưu, không cần GPU
    print(f"RUN_TRAIN = False: vẽ lại từ {HISTORY_PATH} ({len(saved['history'])} epoch)")
    plot_history(saved["history"], saved.get("teacher"), saved["best_epoch"], BIO_DIR / "curves.png")
else:
    print(f"RUN_TRAIN = False và chưa có {HISTORY_PATH}: đặt RUN_TRAIN = True trên GPU để huấn luyện và vẽ.")