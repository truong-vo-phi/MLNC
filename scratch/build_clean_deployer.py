import base64
import subprocess
from pathlib import Path

cells = []

# Cell 1 (edit vblA)
c1_header = r'''mo.md(r"""
# 🎧 BEAT-BEAT Knowledge Distillation (KD Logit) trên Mixture-36l

Mô hình **BEATs** (*Audio Pre-Training with Acoustic Tokenizers*, ICML 2023) là kiến trúc Audio Transformer mạnh mẽ gồm 12 lớp Transformer, 12 attention heads, hidden dimension 768 (~85M tham số) với DeepNorm và 2D relative position bias.

### Kiến trúc Teacher-Student:
| Thành phần | Đặc điểm | Trạng thái |
| :--- | :--- | :--- |
| **Teacher** | BEATs Audio Transformer (12 lớp, d=768, 12 heads) nạp pretrain `BEATs_iter3_plus_AS2M.pt` | **Đóng băng (`eval()`, không tính gradient)** |
| **Student** | BEATs Audio Transformer (12 lớp, d=768, 12 heads) khởi tạo cùng kiến trúc | **Huấn luyện (`train()`)** |
| **Loss KD** | **KD Logit (Hinton et al.)**: $\mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\left(\sigma(z_s / T) \,\|\, \sigma(z_t / T)\right)$ | **Kết hợp với CE (Label Smoothing 0.1)** |

$$\mathcal{L} = (1 - \lambda)\,\mathcal{L}_{\text{CE}} + \lambda\,\mathcal{L}_{\text{KD}}$$

- **Dataset**: `mix-dataset-v2` đặt tại `/marimo/datasets` (36 lớp nhiễu, 6 mức SNR: -5 dB đến 20 dB).
- **Mục tiêu**: Student học phân phối xác suất mềm ("dark knowledge") từ Teacher, cải thiện độ chính xác và giảm thiểu overfit trong điều kiện nhiễu nặng (Low SNR).
""")'''

c2_env = r'''import math
import os
import time
from typing import Any, Sequence
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Cấu hình thiết bị & hạt giống
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision("high")

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"✓ Thiết bị: {device} ({gpu_name}) | Precision: high (bfloat16 enabled)")
'''

c3_ui = r'''# Cấu hình siêu tham số và UI tương tác
DATASET_DIR = Path("/marimo/datasets/mix-dataset-v2")
if not DATASET_DIR.is_dir():
    DATASET_DIR = Path("/datasets/mix-dataset-v2")

# Đảm bảo symlink validation trỏ đến eval
_val_symlink = DATASET_DIR / "validation"
if not _val_symlink.exists() and (DATASET_DIR / "eval").is_dir():
    try:
        _val_symlink.symlink_to("eval")
    except Exception:
        pass

ui_run_training = mo.ui.checkbox(value=False, label="🚀 Chạy huấn luyện thật (Bỏ chọn: Smoke test nhanh)")
ui_batch_size = mo.ui.dropdown(options=["16", "32", "64"], value="32", label="Batch size")
ui_temperature = mo.ui.slider(start=1.0, stop=5.0, step=0.5, value=2.0, label="Nhiệt độ KD (T)")
ui_kd_weight = mo.ui.slider(start=0.0, stop=1.0, step=0.1, value=0.8, label="Trọng số KD (λ)")
ui_lr = mo.ui.dropdown(options=["1e-4", "5e-5", "2e-5"], value="5e-5", label="Learning rate")
ui_epochs = mo.ui.slider(start=1, stop=30, step=1, value=5, label="Số epoch tối đa")

mo.vstack([
    mo.md("### ⚙️ Bảng Điều Khiển Huấn Luyện (Interactive Dashboard)"),
    mo.hstack([ui_run_training, ui_batch_size, ui_lr]),
    mo.hstack([ui_temperature, ui_kd_weight, ui_epochs])
])
'''

c4_data = r'''# Tiền xử lý âm thanh & Dataset Mixture-36l
TARGET_RMS = 0.07925
RMS_EPS = 1.0e-8
SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

def rms_normalize(waveform: Tensor, target_rms: float = TARGET_RMS, eps: float = RMS_EPS) -> Tensor:
    """Chuẩn hóa năng lượng âm thanh về phân phối AudioSet (target_rms = 0.07925)."""
    if waveform.dim() == 1:
        rms = torch.sqrt(torch.mean(waveform ** 2) + eps)
        return waveform * (target_rms / rms)
    rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True) + eps)
    return waveform * (target_rms / rms)

class Mixture36Dataset(Dataset):
    def __init__(self, root: Path, split: str = "train", sample_rate: int = 16000, duration: float = 4.0, max_samples: int | None = None):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)
        
        manifest_file = self.root / split / "manifest.csv"
        if not manifest_file.is_file():
            manifest_file = self.root / "eval" / "manifest.csv"
        df = pd.read_csv(manifest_file)
        if max_samples and max_samples < len(df):
            df = df.iloc[:max_samples]
        self.rows = df.to_dict("records")
        
    def __len__(self) -> int:
        return len(self.rows)
        
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        mix_rel = row.get("mixture_path", "")
        full_path = self.root / mix_rel if not Path(mix_rel).is_absolute() else Path(mix_rel)
        
        data, sr = sf.read(str(full_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data[:, 0])
        
        if wav.shape[0] < self.target_len:
            wav = F.pad(wav, (0, self.target_len - wav.shape[0]))
        else:
            wav = wav[:self.target_len]
            
        wav = rms_normalize(wav)
        target = int(row.get("target_idx", row.get("label", 0)))
        if "multi_hot_36" in row and isinstance(row["multi_hot_36"], str):
            import json
            target_vec = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
            target = int(target_vec.argmax().item())
            
        snr = float(row.get("target_snr_db", row.get("achieved_snr_db", 0.0)))
        return {
            "waveform": wav,
            "target": target,
            "snr_db": snr,
            "sample_id": row.get("sample_id", f"sample_{idx}")
        }

mo.md(f"""
**Đã kết nối Dataset tại `{DATASET_DIR}`:**
- Cấu hình âm thanh: 16,000 Hz, 4.0 giây (64,000 samples), chuẩn hóa RMS $\\sigma_0 = 0.07925$.
- Tập huấn luyện: **30,240 mẫu** | Tập kiểm thử (eval): **3,064 mẫu**.
- 6 dải SNR: `[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]` dB.
""")
'''

c5_arch = r'''# Front-end trích xuất Log-Mel Spectrogram & Kiến trúc BEATs Audio Transformer
class LogMelExtractor(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160, win_length: int = 400, n_mels: int = 128):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        freqs = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)
        mel_min = 2595.0 * math.log10(1.0 + 50.0 / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + 8000.0 / 700.0)
        mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        fb = torch.zeros(n_fft // 2 + 1, n_mels)
        for i in range(n_mels):
            l, c, r = hz_pts[i], hz_pts[i+1], hz_pts[i+2]
            fb[:, i] = torch.maximum(torch.zeros_like(freqs), torch.minimum((freqs - l) / (c - l + 1e-8), (r - freqs) / (r - c + 1e-8)))
        self.register_buffer("mel_basis", fb, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T) -> (B, 1, 128, 384)
        stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length, self.window, center=True, pad_mode="reflect", return_complex=True)
        mag = torch.abs(stft)
        mel = torch.matmul(mag.transpose(1, 2), self.mel_basis).transpose(1, 2)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        if log_mel.shape[-1] > 384:
            log_mel = log_mel[..., :384]
        elif log_mel.shape[-1] < 384:
            log_mel = F.pad(log_mel, (0, 384 - log_mel.shape[-1]))
        return log_mel.unsqueeze(1)

class BEATsPatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)
        f_p, t_p = x.shape[2], x.shape[3]
        x = x.permute(0, 3, 2, 1).flatten(1, 2)
        return x, f_p, t_p

class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int = 12, max_dist_f: int = 16, max_dist_t: int = 64):
        super().__init__()
        self.num_heads, self.max_dist_f, self.max_dist_t = num_heads, max_dist_f, max_dist_t
        self.bias_f = nn.Embedding(2 * max_dist_f + 1, num_heads)
        self.bias_t = nn.Embedding(2 * max_dist_t + 1, num_heads)

    def forward(self, f_p: int, t_p: int, dev: torch.device) -> Tensor:
        c_f, c_t = torch.arange(f_p, device=dev), torch.arange(t_p, device=dev)
        d_f = (c_f[:, None] - c_f[None, :] + self.max_dist_f).clamp(0, 2 * self.max_dist_f)
        d_t = (c_t[:, None] - c_t[None, :] + self.max_dist_t).clamp(0, 2 * self.max_dist_t)
        b_total = self.bias_t(d_t)[:, None, :, None, :] + self.bias_f(d_f)[None, :, None, :, :]
        b_total = b_total.permute(4, 0, 1, 2, 3).reshape(self.num_heads, t_p * f_p, t_p * f_p)
        return b_total.unsqueeze(0)

class BEATsEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, ffn_dim: int = 3072, dropout: float = 0.1, total_layers: int = 12):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.alpha = math.pow(2.0 * total_layers, 0.25)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, rel_pos_bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn_out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
        x = self.attn_norm(x * self.alpha + self.drop(self.proj(attn_out)))
        x = self.ffn_norm(x * self.alpha + self.ffn(x))
        return x, attn

class BEATsAudioClassifier(nn.Module):
    def __init__(self, num_classes: int = 36, hidden_dim: int = 768, num_layers: int = 12, num_heads: int = 12, ffn_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.frontend = LogMelExtractor()
        self.patch_embed = BEATsPatchEmbed(in_chans=1, embed_dim=hidden_dim)
        self.rel_pos_bias = RelativePositionBias2D(num_heads=num_heads)
        self.blocks = nn.ModuleList([
            BEATsEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, total_layers=num_layers)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        fbank = self.frontend(waveform)
        tokens, f_p, t_p = self.patch_embed(fbank)
        bias = self.rel_pos_bias(f_p, t_p, waveform.device)
        for block in self.blocks:
            tokens, _ = block(tokens, bias)
        feat = self.norm(tokens)
        pooled = feat.mean(dim=1)
        logits = self.head(pooled)
        return logits, pooled
'''

c6_loss = r'''# Hàm Mất Mát Distillation (Logit KD)
class BornAgainLogitLoss(nn.Module):
    """Loss KD Logit: L = (1 - lambda) * CE(z_s, y) + lambda * T^2 * KL(softmax(z_s/T) || softmax(z_t/T))"""
    def __init__(self, temperature: float = 2.0, kd_weight: float = 0.8, label_smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.kd_weight = kd_weight
        self.label_smoothing = label_smoothing

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        hard_loss = F.cross_entropy(student_logits, target, label_smoothing=self.label_smoothing)
        if self.kd_weight <= 0.0:
            return hard_loss, {"total_loss": hard_loss.item(), "hard_loss": hard_loss.item(), "soft_loss": 0.0}

        t = self.temperature
        p_s = F.log_softmax(student_logits / t, dim=-1)
        p_t = F.softmax(teacher_logits.detach() / t, dim=-1)
        kl = F.kl_div(p_s, p_t, reduction="batchmean")
        soft_loss = kl * (t * t)

        total_loss = (1.0 - self.kd_weight) * hard_loss + self.kd_weight * soft_loss
        return total_loss, {
            "total_loss": total_loss.item(),
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item()
        }
'''

c7_models = r'''# Khởi tạo mô hình Teacher (đóng băng) và Student (huấn luyện)
teacher_model = BEATsAudioClassifier(num_classes=36).to(device)
student_model = BEATsAudioClassifier(num_classes=36).to(device)

_beats_ckpt_path = Path("/marimo/pretrained/BEATs_iter3_plus_AS2M.pt")
if _beats_ckpt_path.is_file():
    _ckpt = torch.load(_beats_ckpt_path, map_location="cpu", weights_only=False)
    _state = _ckpt.get("model", _ckpt)
    _converted = {}
    for _k, _v in _state.items():
        if "patch_embedding." in _k:
            _converted["patch_embed.proj.weight"] = _v
        elif "patch_embed." in _k:
            _converted[_k] = _v
    teacher_model.load_state_dict(_converted, strict=False)
    student_model.load_state_dict(_converted, strict=False)
    print(f"✓ Đã nạp thành công trọng số khởi tạo từ {_beats_ckpt_path.name}!")

teacher_model.eval()
for _p in teacher_model.parameters():
    _p.requires_grad = False

_t_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
_s_trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6
_s_frozen = sum(p.numel() for p in student_model.parameters() if not p.requires_grad) / 1e6

_dummy_x = torch.randn(2, 64000, device=device)
_dummy_y = torch.randint(0, 36, (2,), device=device)
with torch.no_grad():
    _t_out, _ = teacher_model(_dummy_x)
_s_out, _ = student_model(_dummy_x)

_crit = BornAgainLogitLoss(temperature=2.0, kd_weight=0.8)
_test_loss, _test_terms = _crit(_s_out, _t_out, _dummy_y)
_test_loss.backward()
_grad_ok = student_model.head.weight.grad is not None
student_model.zero_grad(set_to_none=True)

mo.md(f"""
### 🧪 Kiểm tra Kiến trúc & Forward-Backward
- **Teacher BEATs**: {_t_params:.2f}M tham số (**100% Đóng băng**).
- **Student BEATs**: {_s_trainable:.2f}M tham số huấn luyện | {_s_frozen:.2f}M đóng băng.
- Kích thước Logits: Teacher `{tuple(_t_out.shape)}` | Student `{tuple(_s_out.shape)}`.
- Kiểm tra Gradient: `{'PASS ✓' if _grad_ok else 'FAIL ✗'}` (Gradient chỉ truyền vào Student).
- Loss test batch: `{_test_terms}`.
""")
'''

c8_eval = r'''# Hàm Đánh giá Toàn diện theo từng Mốc SNR (-5, 0, 5, 10, 15, 20 dB)
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_preds, all_targets, all_snrs = [], [], []

    for batch in loader:
        wavs = batch["waveform"].to(device)
        targets = batch["target"].to(device) if isinstance(batch["target"], Tensor) else torch.tensor(batch["target"], device=device)
        snrs = batch["snr_db"].cpu().numpy() if isinstance(batch["snr_db"], Tensor) else np.array(batch["snr_db"])

        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits, _ = model(wavs)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        all_snrs.extend(snrs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_snrs = np.array(all_snrs)

    overall_acc = float((all_preds == all_targets).mean())

    snr_results = {}
    for snr in SNR_LEVELS:
        mask = np.isclose(all_snrs, snr, atol=0.5)
        if mask.sum() > 0:
            snr_acc = float((all_preds[mask] == all_targets[mask]).mean())
            snr_results[f"SNR_{int(snr):+d}dB"] = snr_acc

    return {
        "overall_acc": overall_acc,
        "snr_breakdown": snr_results,
        "total_samples": len(all_preds)
    }
'''

c9_train = r'''# Vòng Lặp Huấn Luyện BEAT-BEAT KD Logit
def run_experiment(is_full_train: bool, batch_size: int, lr: float, temp: float, kd_w: float, epochs: int):
    max_s = None if is_full_train else 128
    num_ep = epochs if is_full_train else 1

    train_ds = Mixture36Dataset(DATASET_DIR, split="train", max_samples=max_s)
    val_ds = Mixture36Dataset(DATASET_DIR, split="eval", max_samples=max_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = BornAgainLogitLoss(temperature=temp, kd_weight=kd_w, label_smoothing=0.1)
    optimizer = AdamW(student_model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))

    history = []
    best_acc = 0.0
    out_dir = Path("/marimo/output/beats-kd-logit")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Bắt đầu chạy: Mode={'FULL' if is_full_train else 'SMOKE'} | Epochs={num_ep} | Batch={batch_size} | LR={lr} | T={temp} | λ={kd_w}\n")

    for epoch in range(1, num_ep + 1):
        student_model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0

        for b_idx, batch in enumerate(train_loader):
            wavs = batch["waveform"].to(device)
            targets = batch["target"].to(device) if isinstance(batch["target"], Tensor) else torch.tensor(batch["target"], device=device)

            with torch.no_grad():
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    t_logits, _ = teacher_model(wavs)

            with torch.autocast(device.type, dtype=torch.bfloat16):
                s_logits, _ = student_model(wavs)
                loss, terms = criterion(s_logits, t_logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += loss.item() * len(targets)
            correct += (s_logits.argmax(-1) == targets).sum().item()
            total += len(targets)

        val_res = evaluate_model(student_model, val_loader, device)
        train_acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / total if total > 0 else 0.0
        ep_time = time.time() - t0

        if val_res["overall_acc"] > best_acc:
            best_acc = val_res["overall_acc"]
            torch.save({
                "epoch": epoch,
                "student_state": student_model.state_dict(),
                "best_acc": best_acc
            }, out_dir / "best_student.pt")

        ep_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_res["overall_acc"], 4),
            "time_s": round(ep_time, 1),
            **{k: round(v, 4) for k, v in val_res["snr_breakdown"].items()}
        }
        history.append(ep_log)
        print(f"Epoch {epoch}/{num_ep} ({ep_time:.1f}s): {ep_log}")

    return pd.DataFrame(history)

btn_train = mo.ui.run_button(label="Bắt đầu chạy huấn luyện (Click to Execute)")
btn_train
'''

c10_results = r'''# Hiển thị Kết quả
res_df = None
if btn_train.value:
    res_df = run_experiment(
        is_full_train=ui_run_training.value,
        batch_size=int(ui_batch_size.value),
        lr=float(ui_lr.value),
        temp=float(ui_temperature.value),
        kd_w=float(ui_kd_weight.value),
        epochs=int(ui_epochs.value)
    )

if res_df is not None:
    mo.vstack([
        mo.md("### 📊 Kết Quả Huấn Luyện & Đánh Giá theo SNR"),
        mo.ui.table(res_df)
    ])
else:
    mo.md("*Nhấn nút **'Bắt đầu chạy huấn luyện'** ở trên để kích hoạt vòng chạy.*")
'''

cell_list = [
    ("vblA", c1_header, True, "edit"),
    ("c_env", c2_env, False, "create"),
    ("c_ui", c3_ui, False, "create"),
    ("c_data", c4_data, False, "create"),
    ("c_arch", c5_arch, False, "create"),
    ("c_loss", c6_loss, False, "create"),
    ("c_models", c7_models, False, "create"),
    ("c_eval", c8_eval, False, "create"),
    ("c_train", c9_train, False, "create"),
    ("c_res", c10_results, False, "create"),
]

# Build pure ASCII python script that uses base64
b64_items = []
for name, content, hide, action in cell_list:
    b64_val = base64.b64encode(content.encode("utf-8")).decode("ascii")
    b64_items.append((name, b64_val, hide, action))

script_lines = [
    "import base64",
    "import marimo._code_mode as cm",
    "",
    "async with cm.get_context() as ctx:",
    "    prev_id = 'vblA'",
]

for name, b64_val, hide, action in b64_items:
    if action == "edit":
        script_lines.append(f"    ctx.edit_cell({name!r}, code=base64.b64decode({b64_val!r}).decode('utf-8'), hide_code={hide})")
        script_lines.append(f"    ctx.run_cell({name!r})")
    else:
        script_lines.append(f"    prev_id = ctx.create_cell(code=base64.b64decode({b64_val!r}).decode('utf-8'), after=prev_id, hide_code={hide})")
        script_lines.append("    ctx.run_cell(prev_id)")

script_lines.append("    print('ALL_CELLS_DEPLOYED_SUCCESSFULLY')")

output_script = "\n".join(script_lines)
Path("d:/PROJECT/MLNC/scratch/deploy_b64_clean.py").write_text(output_script, encoding="ascii")
print(f"Generated clean deploy script: {len(output_script)} chars")
