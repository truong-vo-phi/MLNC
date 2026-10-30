from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml
from torch import Tensor

from src.core import BaseModel

from .config import StudentConfig
from .frontend import LogMel, LogMelConfig, SpecAugment, SpecAugmentConfig
from .vit_bias import ViTSBias, load_ced_tiny


class StudentOutput(NamedTuple):
    logits: Tensor


class StudentHiddenOutput(NamedTuple):
    logits: Tensor
    hidden: tuple[Tensor, ...]


class CEDStudent(BaseModel):
    def __init__(self, config: StudentConfig, frontend: LogMel, spec_augment: SpecAugment) -> None:
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
        )
        self.init_report = self._initialize()

    @classmethod
    def from_yaml(cls, path: str | Path) -> CEDStudent:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        student = StudentConfig.from_dict(raw.get(StudentConfig.section, {}))
        frontend = LogMelConfig.from_dict(raw.get(LogMelConfig.section, {}))
        spec_augment = SpecAugmentConfig.from_dict(raw.get(SpecAugmentConfig.section, {}))
        return cls(student, LogMel(**frontend.to_dict()), SpecAugment(**spec_augment.to_dict()))

    def _initialize(self) -> dict[str, int]:
        report: dict[str, int] = {}
        if self.config.init == "ced-tiny":
            report = load_ced_tiny(self.vit, self.config.ced_repo, self.config.ced_path)
        if self.config.bias_init == "zero":
            self.vit.attentions[0].relative_attention_bias.weight.data.zero_()
        return report

    def forward(self, wave: Tensor) -> StudentOutput:
        with torch.autocast(wave.device.type, enabled=False):
            features = self.frontend(wave.float())
        return StudentOutput(logits=self.vit(self.spec_augment(features)))

    def predict_proba(self, wave: Tensor) -> Tensor:
        return self(wave).logits.softmax(dim=-1)

    def forward_with_hidden(self, wave: Tensor, layers: tuple[int, ...] = ()) -> StudentHiddenOutput:
        captured: dict[int, Tensor] = {}

        def capture(index: int):
            def hook(_module, _inputs, output):
                captured[index] = output

            return hook

        handles = [self.vit.vit.blocks[layer - 1].register_forward_hook(capture(layer)) for layer in layers]
        try:
            logits = self(wave).logits
        finally:
            for handle in handles:
                handle.remove()
        hidden = tuple(captured[layer][:, 1:] for layer in layers)
        return StudentHiddenOutput(logits=logits, hidden=hidden)

    def set_record(self, enabled: bool) -> None:
        self.vit.set_record(enabled)

    def attention_caches(self) -> list[dict[str, Tensor]]:
        return [attention.cache for attention in self.vit.attentions]

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def describe_params(self) -> dict[str, int]:
        return self.vit.describe_params()
