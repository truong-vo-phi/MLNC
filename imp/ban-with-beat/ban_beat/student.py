from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.model.beat import BEATClassifier, BEATOutput

from .frontend import apply_front_end
from .teacher import BANTeacher


class BANStudent(nn.Module):
    def __init__(
        self,
        classifier: BEATClassifier,
        separator: nn.Module | None = None,
        input_rms: float | None = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.separator = separator
        self.input_rms = input_rms
        self.config = classifier.config
        if self.separator is not None:
            self.separator.eval()
            for parameter in self.separator.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> BANStudent:
        super().train(mode)
        if self.separator is not None:
            self.separator.eval()
        return self

    def as_teacher(self) -> BANTeacher:
        return BANTeacher(self.classifier, self.separator, self.input_rms).freeze()

    def forward(self, waveform: Tensor, layers: Sequence[int] = ()) -> BEATOutput:
        with torch.no_grad():
            features = apply_front_end(waveform, self.separator, self.input_rms)
        return self.classifier(features, layers=layers)
