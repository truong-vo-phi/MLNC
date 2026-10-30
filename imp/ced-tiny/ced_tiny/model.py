from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml
from torch import Tensor

from src.core import BaseModel
from src.model.frcrn import FRCRNSeparator

from .config import CEDFRCRNConfig, CEDTinyConfig, FrontendConfig, SpecAugmentConfig, resolve_path
from .frontend import LogMel, SpecAugment
from .vit_bias import ViTSBias, load_ced_tiny

RMS_EPS = 1e-8


def rms_normalize(waveform: Tensor, target_rms: float) -> Tensor:
    rms = waveform.square().mean(dim=-1).sqrt()
    return waveform * (target_rms / (rms + RMS_EPS))[:, None]


class CEDOutput(NamedTuple):
    logits: Tensor


class CEDTiny(BaseModel):
    def __init__(self, config: CEDTinyConfig, frontend: LogMel, spec_augment: SpecAugment) -> None:
        super().__init__()
        self.config = config
        self.frontend = frontend
        self.spec_augment = spec_augment
        self.vit = ViTSBias(
            num_classes=config.num_classes,
            freq_bins=config.freq_bins,
            frames=config.frames,
            pretrained=config.init == "imagenet",
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            drop_path=config.drop_path,
            post_ln=config.post_ln,
            pre_ln=config.pre_ln,
            arch=config.arch,
            bias_init=config.bias_init,
            use_bias=config.use_bias,
            time_major=config.time_major,
            hybrid_pool=config.hybrid_pool,
        )
        self.init_report = self._initialize()

    @classmethod
    def from_yaml(cls, path: str | Path) -> CEDTiny:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        ced_cfg = CEDTinyConfig.from_dict(raw.get(CEDTinyConfig.section, {}))
        frontend_cfg = FrontendConfig.from_dict(raw.get(FrontendConfig.section, {}))
        spec_cfg = SpecAugmentConfig.from_dict(raw.get(SpecAugmentConfig.section, {}))
        return cls(ced_cfg, LogMel(**frontend_cfg.to_dict()), SpecAugment(**spec_cfg.to_dict()))

    def _initialize(self) -> dict[str, int]:
        report: dict[str, int] = {}
        if self.config.init == "ced-tiny":
            report = load_ced_tiny(self.vit, self.config.ced_repo, self.config.ced_path)
        return report

    def forward(self, wave: Tensor) -> CEDOutput:
        with torch.autocast(wave.device.type, enabled=False):
            features = self.frontend(wave.float())
        return CEDOutput(logits=self.vit(self.spec_augment(features)))

    def predict_proba(self, wave: Tensor) -> Tensor:
        return self(wave).logits.softmax(dim=-1)

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def describe_params(self) -> dict[str, int]:
        return self.vit.describe_params()


class FRCRNCEDPipeline(BaseModel):
    def __init__(
        self,
        separator: FRCRNSeparator,
        classifier: CEDTiny,
        target_rms: float = 0.07925,
        freeze_separator: bool = True,
    ) -> None:
        super().__init__()
        self.separator = separator
        self.classifier = classifier
        self.target_rms = target_rms
        if freeze_separator:
            self.separator.freeze()

    @classmethod
    def from_yaml(cls, path: str | Path) -> FRCRNCEDPipeline:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        frcrn_cfg_path = resolve_path(raw.get("frcrn_config", "config/frcrn.yaml"))
        separator = FRCRNSeparator.from_yaml(frcrn_cfg_path)
        classifier = CEDTiny.from_yaml(path)
        frcrn_dict = raw.get(CEDFRCRNConfig.section, {})
        target_rms = float(frcrn_dict.get("target_rms", 0.07925))
        freeze = bool(frcrn_dict.get("freeze", True))
        return cls(separator, classifier, target_rms=target_rms, freeze_separator=freeze)

    def forward(self, mixture: Tensor) -> CEDOutput:
        with torch.no_grad():
            noise_hat = self.separator(mixture)
        normalized = rms_normalize(noise_hat, self.target_rms)
        return self.classifier(normalized)

    @torch.no_grad()
    def predict_proba(self, mixture: Tensor) -> Tensor:
        noise_hat = self.separator(mixture)
        normalized = rms_normalize(noise_hat, self.target_rms)
        return self.classifier.predict_proba(normalized)

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        if "classifier" in checkpoint:
            return checkpoint["classifier"]
        if "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint
