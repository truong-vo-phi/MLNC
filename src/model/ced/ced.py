from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import yaml
from safetensors.torch import load_file
from torch import Tensor

from src.core import BaseModel

from .backbone import CEDBackbone
from .config import CEDConfig
from .frontend import MelFrontend
from .head import ClassifierHead

_ENCODER_PREFIX = "encoder."


class CEDOutput(NamedTuple):
    logits: Tensor
    features: Tensor


class CEDClassifier(BaseModel):
    expected_missing_prefixes: ClassVar[tuple[str, ...]] = ("head.", "frontend.")

    def __init__(self, config: CEDConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config
        self.frontend = MelFrontend(n_mels=config.n_mels, **(frontend_config or {}))
        self.backbone = CEDBackbone(config)
        self.head = ClassifierHead(config.embed_dim, config.num_classes)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CEDClassifier:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(CEDConfig.from_dict(raw[CEDConfig.section]), raw.get("frontend"))

    def forward(self, x: Tensor, is_waveform: bool = True) -> CEDOutput:
        spectrogram = self.frontend(x) if is_waveform else x
        features = self.backbone(spectrogram)
        return CEDOutput(logits=self.head(features), features=features)

    def read_checkpoint(self, path: str | Path) -> dict[str, Any]:
        return load_file(path)

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return {
            f"backbone.{key.removeprefix(_ENCODER_PREFIX)}": value
            for key, value in checkpoint.items()
            if key.startswith(_ENCODER_PREFIX)
        }

    def predict_proba(self, x: Tensor, is_waveform: bool = True) -> Tensor:
        return self(x, is_waveform).logits.softmax(dim=-1)
