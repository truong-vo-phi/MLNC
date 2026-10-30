from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=30.0,
)

code = """
import torch
ckpt = torch.load('/marimo/output/high_expert/last_best_checkpoint.pt', map_location='cpu')
print('High Expert Best Epoch:', ckpt.get('epoch'))
acc = ckpt.get('best_acc')
print(f'High Expert Best SNR 20 Val Acc: {acc * 100:.2f}%')
"""

out = client.run(code)
sys.stdout.buffer.write(out.encode("utf-8"))
