from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from src.core import BaseModel
from src.model.beat import BEATClassifier, BEATOutput
from src.model.frcrn import FRCRNSeparator

RMS_EPS = 1e-8


def rms_normalize(waveform: Tensor, target_rms: float) -> Tensor:
    rms = waveform.square().mean(dim=-1).sqrt()
    return waveform * (target_rms / (rms + RMS_EPS))[:, None]


class FRCRNBEATPipeline(BaseModel):
    def __init__(
        self,
        separator: FRCRNSeparator,
        classifier: BEATClassifier,
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
    def from_yaml(cls, path: str | Path) -> FRCRNBEATPipeline:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        repo_root = Path(path).resolve().parents[2]
        model_cfg = Path(raw.get("model_config", "config/beat.yaml"))
        frcrn_cfg = Path(raw.get("frcrn_config", "config/frcrn.yaml"))
        if not model_cfg.is_absolute():
            model_cfg = repo_root / model_cfg
        if not frcrn_cfg.is_absolute():
            frcrn_cfg = repo_root / frcrn_cfg
        separator = FRCRNSeparator.from_yaml(frcrn_cfg)
        classifier = BEATClassifier.from_yaml(model_cfg)
        target_rms = float(raw.get("frcrn", {}).get("target_rms", 0.07925))
        return cls(separator, classifier, target_rms=target_rms)

    def forward(self, mixture: Tensor) -> BEATOutput:
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
