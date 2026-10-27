from __future__ import annotations

from typing import Any

from torch import Tensor

PANNS_NUM_BLOCKS = 4
# Thứ tự lớp trong ConvBlock.layers: conv, bn, relu, conv, bn, relu, pool, dropout
PANNS_BLOCK_LAYERS = {"conv1": 0, "bn1": 1, "conv2": 3, "bn2": 4}


def convert_panns_cnn10(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    """Map a PANNs Cnn10 checkpoint onto CNN10 keys, dropping bn0, the mel buffers and the 527-class head."""
    state = checkpoint.get("model", checkpoint)
    converted: dict[str, Tensor] = {}
    for index in range(PANNS_NUM_BLOCKS):
        for name, layer in PANNS_BLOCK_LAYERS.items():
            prefix = f"conv_block{index + 1}.{name}"
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                if f"{prefix}.{suffix}" in state:
                    converted[f"blocks.{index}.layers.{layer}.{suffix}"] = state[f"{prefix}.{suffix}"]
    for suffix in ("weight", "bias"):
        converted[f"fc.{suffix}"] = state[f"fc1.{suffix}"]
    return converted
