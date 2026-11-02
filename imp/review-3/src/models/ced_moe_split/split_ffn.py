"""
Split-FFN MoE Module with Shared and Routed Experts for CED-Tiny Blocks 10 & 11.

Architecture:
    y = 0.5 * FFN_shared(x) + 0.5 * [(1 - alpha) * FFN_low(x) + alpha * FFN_high(x)]

Total weight:
    0.5 + 0.5 * (1 - alpha + alpha) = 1.0 (Strictly normalized convex combination).

Warm-Start Identity Initialization:
    At step 0, when ffn_shared, ffn_low, ffn_high are cloned from pretrained base MLP:
    y = 0.5 * M(x) + 0.5 * [(1 - alpha) * M(x) + alpha * M(x)] = M(x)
    Output is mathematically identical to original CED-Tiny for ALL alpha in (0, 1).
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch import Tensor


class SplitMoEFFN(nn.Module):
    """
    Split-FFN containing 1 Shared Expert + 2 Routed Experts (Low-SNR & High-SNR).
    """
    def __init__(self, base_mlp: nn.Module) -> None:
        super().__init__()
        self.ffn_shared = copy.deepcopy(base_mlp)
        self.ffn_low = copy.deepcopy(base_mlp)
        self.ffn_high = copy.deepcopy(base_mlp)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        """
        Args:
            x: [B, Tokens, C] (e.g. [B, 97, 192])
            alpha: [B, 1] or [B, 1, 1] continuous soft gate weight
        Returns:
            out: [B, Tokens, C]
        """
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]

        out_shared = self.ffn_shared(x)
        out_low = self.ffn_low(x)
        out_high = self.ffn_high(x)

        # Normalized convex combination
        routed_out = (1.0 - alpha) * out_low + alpha * out_high
        return 0.5 * out_shared + 0.5 * routed_out

