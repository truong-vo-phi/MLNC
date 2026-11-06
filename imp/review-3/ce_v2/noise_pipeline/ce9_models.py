"""Nine genuine pretrained model adapters returning raw 36-class CE logits."""
from __future__ import annotations

import torch
import torchvision  # Initialize operator registration outside upstream namespace-package aliases.

# Scoped compatibility patch for torchaudio on Python 3.13 / PyTorch 2.11
if hasattr(torch, "_ops") and hasattr(torch._ops, "_OpNamespace"):
    _orig_op_getattr = torch._ops._OpNamespace.__getattr__
    def _safe_op_getattr(self, op_name):
        try:
            return _orig_op_getattr(self, op_name)
        except AttributeError:
            if getattr(self, "name", None) == "torchaudio":
                return lambda *args, **kwargs: None
            raise
    torch._ops._OpNamespace.__getattr__ = _safe_op_getattr

from contextlib import contextmanager, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import torchvision  # Initialize operator registration outside upstream namespace-package aliases.

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("panns_cnn10", "ced_tiny", "ced_mini", "ced_small", "efficientat_mn10",
               "vit_small", "mobilevit_s", "mobilenet_v3_large")
HF_REVISIONS = {"ced_tiny": "ace276d29dd0bb3f3517b0fa8cf300738c409019",
                "ced_mini": "26c3ebcae85d4330f4fc26763f029539a3afcda0",
                "ced_small": "06bb40c5ec089e96867ebc5246be02441f4a71e4"}
TIMM_NAMES = {"vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
              "mobilevit_s": "mobilevit_s.cvnets_in1k",
              "mobilenet_v3_large": "mobilenetv3_large_100.ra_in1k"}


def beats_checkpoint() -> Path:
    paths_file = ROOT / "benchmark_results/team_kd_moe_research_20261102/TEAM_DATA_PATHS.json"
    if paths_file.is_file():
        paths = json.loads(paths_file.read_text(encoding="utf-8"))
        if paths.get("beats_pretrained"):
            return Path(paths["beats_pretrained"])
    return ROOT / "checkpoints/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"


@contextmanager
def vendor_import(path: Path):
    """Isolate upstream repositories that all use top-level `models` imports."""
    prefixes = ("models", "helpers", "pytorch_utils")
    saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in prefixes}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(path))
    previous_directory = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        for k in list(sys.modules):
            if k.split('.')[0] in prefixes:
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(path))


class CEDAdapter(nn.Module):
    def __init__(self, name: str, classes: int, assets: Path):
        super().__init__()
        from safetensors.torch import load_file
        config = json.loads((assets / name / "config.json").read_text())
        kwargs = {k: v for k, v in config.items() if k not in
                  ("architectures", "auto_map", "id2label", "label2id", "model_type", "name", "torch_dtype", "transformers_version")}
        # `logit` pooling is the upstream mean-pooling path before sigmoid.
        kwargs["pooling"] = "logit"
        with vendor_import(ROOT / "third_party/ced_ce9"):
            cls = importlib.import_module("models.audiotransformer").AudioTransformer
            self.model = cls(**kwargs)
        state = load_file(str(assets / name / "model.safetensors"))
        mapped = {}
        for key, value in state.items():
            key = key.removeprefix("encoder.")
            if key.startswith("init_bn."):
                key = key.replace("init_bn.", "init_bn.1.", 1)
            mapped[key] = value
        receipt = nn.Module.load_state_dict(self.model, mapped, strict=False)
        learned = {n for n, _ in self.model.named_parameters()}
        if receipt.unexpected_keys or learned.intersection(receipt.missing_keys):
            raise ValueError(f"Incomplete CED pretrained load: {receipt}")
        self.model.outputlayer[1] = nn.Linear(config["embed_dim"], classes)
        self.receipt = {"domain": "AudioSet", "hf_revision": HF_REVISIONS[name],
                        "loaded_tensors": len(mapped), "missing_buffers": receipt.missing_keys,
                        "pooling": "original mean pooling, raw logits; no CLS/GREP/MoE"}

    def forward(self, wave):
        return self.model(wave.float())


class PANNAdapter(nn.Module):
    def __init__(self, classes: int, assets: Path):
        super().__init__()
        with vendor_import(ROOT / "third_party/panns_ce9/pytorch"):
            cls = importlib.import_module("models").Cnn10
            self.model = cls(32000, 1024, 320, 64, 50, 14000, 527)
        saved = torch.load(assets / "Cnn10_mAP=0.380.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.fc_audioset = nn.Linear(512, classes)
        self.model.spec_augmenter = nn.Identity()
        self.resample = torchaudio.transforms.Resample(16000, 32000)
        self.receipt = {"domain": "AudioSet", "model": "official PANNs Cnn10", "load": "strict before new head"}

    def forward(self, wave):
        # Reproduce the original Cnn10 path and return its classifier logits.
        m = self.model
        with torch.autocast(wave.device.type, enabled=False):
            wave = self.resample(wave.float())
            x = m.logmel_extractor(m.spectrogram_extractor(wave))
        x = m.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (m.conv_block1, m.conv_block2, m.conv_block3, m.conv_block4):
            x = block(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=.2, training=self.training)
        x = x.mean(3)
        x = x.amax(2) + x.mean(2)
        x = F.dropout(x, p=.5, training=self.training)
        return m.fc_audioset(F.relu_(m.fc1(x)))


class EfficientAdapter(nn.Module):
    def __init__(self, classes: int, assets: Path):
        super().__init__()
        with vendor_import(ROOT / "third_party/efficientat_ce9"):
            module = importlib.import_module("models.mn.model")
            with redirect_stdout(io.StringIO()):
                self.model = module.get_model(num_classes=527, width_mult=1., pretrained_name=None)
            frontend = importlib.import_module("models.preprocess").AugmentMelSTFT
            self.frontend = frontend(n_mels=128, sr=32000, win_length=800, hopsize=320,
                                     n_fft=1024, freqm=0, timem=0, fmin=0, fmax=15000,
                                     fmin_aug_range=1, fmax_aug_range=1)
        state = torch.load(assets / "mn10_as_mAP_471.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.classifier[5] = nn.Linear(1280, classes)
        self.resample = torchaudio.transforms.Resample(16000, 32000)
        self.receipt = {"domain": "AudioSet", "model": "EfficientAT mn10_as width 1.0", "load": "strict before new head"}

    def forward(self, wave):
        with torch.autocast(wave.device.type, enabled=False):
            spec = self.frontend(self.resample(wave.float())).unsqueeze(1)
        return self.model(spec)[0]


class VisionAdapter(nn.Module):
    def __init__(self, name: str, classes: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.model = timm.create_model(TIMM_NAMES[name], pretrained=pretrained, num_classes=classes)
        cfg = self.model.pretrained_cfg
        self.size = tuple(cfg["input_size"][-2:])
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024,
                                                       win_length=400, hop_length=160, n_mels=128)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).reshape(1, 3, 1, 1))
        self.receipt = {"domain": "ImageNet", "timm_name": TIMM_NAMES[name], "image_size": self.size,
                        "pretrained_cfg": cfg, "spectrogram": "16kHz; FFT1024/window400/hop160; 128 Mel; zscore clip +/-3 to [0,1]; replicate RGB; pretrained normalization"}

    def forward(self, wave):
        with torch.autocast(wave.device.type, enabled=False):
            x = self.db(self.mel(wave.float()))
            x = (x-x.mean((1, 2), keepdim=True))/(x.std((1, 2), keepdim=True)+1e-5)
            x = (x.clamp(-3, 3)+3)/6
            x = F.interpolate(x[:, None], self.size, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
            x = (x-self.mean)/self.std
        return self.model(x)


class BEATAdapter(nn.Module):
    def __init__(self, classes: int, assets: Path):
        super().__init__()
        path = ROOT / "third_party/beats"
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from BEATs import BEATs, BEATsConfig
        saved = torch.load(beats_checkpoint(),
                           map_location="cpu", weights_only=True)
        config = dict(saved["cfg"])
        config.update(finetuned_model=False, encoder_layerdrop=0., layer_wise_gradient_decay_ratio=1.)
        self.encoder = BEATs(BEATsConfig(config))
        self.encoder.load_state_dict({k: v for k, v in saved["model"].items() if not k.startswith("predictor.")}, strict=True)
        self.head = nn.Linear(config["encoder_embed_dim"], classes)
        self.receipt = {"domain": "AudioSet", "checkpoint": "official BEATs iter3+ AS2M cpt2", "encoder_layers": 12,
                        "encoder_cfg": config, "head": "new random 36-class head, same policy as other models"}

    def forward(self, wave):
        hidden, _ = self.encoder.extract_features(wave.float())
        return self.head(hidden.mean(1))


def build_model(name: str, assets: Path, classes: int = 36):
    if name.startswith("ced_"):
        return CEDAdapter(name, classes, assets)
    if name == "panns_cnn10":
        return PANNAdapter(classes, assets)
    if name == "efficientat_mn10":
        return EfficientAdapter(classes, assets)
    if name in TIMM_NAMES:
        return VisionAdapter(name, classes)
    if name == "beats12":
        return BEATAdapter(classes, assets)
    raise ValueError(name)
