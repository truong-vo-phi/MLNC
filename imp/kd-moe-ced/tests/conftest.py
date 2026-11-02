"""
Cấu hình fixtures và thiết lập pytest cho kiểm thử kiến trúc Single-Backbone CED MoE.
"""

import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_waveform():
    """Batch gồm 2 đoạn âm thanh giả lập 16kHz trong 4.0 giây (64,000 mẫu)."""
    torch.manual_seed(42)
    return torch.randn(2, 64000)
