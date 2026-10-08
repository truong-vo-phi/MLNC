from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import torch
from torch import Tensor

from .beat import BEATClassifier, BEATConfig, BEATOutput

AUDIOSET_CLASSES = 527


class BEATTeacher(BEATClassifier):
    expected_missing_prefixes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: BEATConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__(replace(config, num_classes=AUDIOSET_CLASSES), frontend_config)
        self.register_buffer("class_indices", torch.arange(AUDIOSET_CLASSES), persistent=False)
        self._mid_to_index: dict[str, int] = {}

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        state = super().convert_checkpoint(checkpoint)
        state["head.fc.weight"] = checkpoint["model"]["predictor.weight"]
        state["head.fc.bias"] = checkpoint["model"]["predictor.bias"]
        self._mid_to_index = {mid: int(index) for index, mid in checkpoint["label_dict"].items()}
        return state

    def select_classes(self, mids: Sequence[str]) -> BEATTeacher:
        if not self._mid_to_index:
            raise ValueError("load_pretrained must be called before select_classes")
        missing = [mid for mid in mids if mid not in self._mid_to_index]
        if missing:
            raise KeyError(f"mids not in the BEATs label dictionary: {missing}")
        self.class_indices = torch.tensor([self._mid_to_index[mid] for mid in mids], device=self.class_indices.device)
        return self

    def forward(self, x: Tensor, is_waveform: bool = True) -> BEATOutput:
        output = super().forward(x, is_waveform)
        return BEATOutput(logits=output.logits[:, self.class_indices], features=output.features)
