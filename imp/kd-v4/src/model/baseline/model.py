from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml
from torch import Tensor

from src.core import BaseModel
from src.model.kdmoe.frontend import LogMel, SpecAugment

from .config import BaselineConfig
from .registry import build_backbone


class BaselineOutput(NamedTuple):
    logits: Tensor
    features: tuple[Tensor, ...] = ()


class BaselineModel(BaseModel):
    def __init__(self, config: BaselineConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config
        frontend = {"n_mels": config.freq_bins, "num_frames": config.num_frames, **(frontend_config or {})}
        if frontend["n_mels"] != config.freq_bins or frontend["num_frames"] != config.num_frames:
            raise ValueError(
                f"frontend (n_mels {frontend['n_mels']}, num_frames {frontend['num_frames']}) does not match "
                f"model (freq_bins {config.freq_bins}, num_frames {config.num_frames})"
            )
        self.frontend = LogMel(**frontend)
        self.spec_augment = SpecAugment(
            freq_masks=config.specaug_freq_masks,
            freq_width=config.specaug_freq_width,
            time_masks=config.specaug_time_masks,
            time_width=config.specaug_time_width,
        )
        self.backbone = build_backbone(config)

    @classmethod
    def from_yaml(cls, path: str | Path, name: str | None = None) -> BaselineModel:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        entries = raw.get("models", {})
        if name is None:
            if len(entries) != 1:
                raise ValueError(f"{path} defines {len(entries)} models, pass name= to pick one of {sorted(entries)}")
            name = next(iter(entries))
        if name not in entries:
            raise KeyError(f"model '{name}' is not in {path}, expected one of {sorted(entries)}")
        values = {**raw[BaselineConfig.section], **entries[name]}
        return cls(BaselineConfig.from_dict(values), raw.get("frontend"))

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return dict(checkpoint.get("model", checkpoint))

    def forward(self, waveform: Tensor, return_features: bool = False) -> BaselineOutput:
        x, _ = self.frontend(waveform)
        x = self.spec_augment(x)
        if not return_features:
            return BaselineOutput(logits=self.backbone(x))
        if not hasattr(self.backbone, "forward_stages"):
            raise ValueError(f"backbone '{self.config.arch}' does not expose stage features")
        logits, features = self.backbone.forward_stages(x)
        return BaselineOutput(logits=logits, features=features)

    def predict_proba(self, waveform: Tensor) -> Tensor:
        return self(waveform).logits.softmax(dim=-1)
