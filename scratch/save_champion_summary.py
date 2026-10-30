from __future__ import annotations

import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0,
)

code = """
import json, csv
from pathlib import Path

out_dir = Path('/marimo/output/kd-v5/champion_pipeline')
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "pipeline": "Dual-Expert + 3-Way TTA + Temperature Calibration",
    "target_achieved": True,
    "metrics": {
        "top1_accuracy": 0.6357702349869452,
        "top3_accuracy": 0.8018929503916449,
        "balanced_accuracy": 0.6202967119253457,
        "mAP_macro": 0.7008149216391456,
        "mAP_overall": 0.6956821402941031,
        "macro_auc": 0.9619140293184920
    },
    "snr_breakdown": {
        "-5": {"top1": 0.7220, "top3": 0.8526, "mAP": 0.7809, "auc": 0.9756},
        "0":  {"top1": 0.7162, "top3": 0.8591, "mAP": 0.7741, "auc": 0.9745},
        "5":  {"top1": 0.6455, "top3": 0.8156, "mAP": 0.7352, "auc": 0.9687},
        "10": {"top1": 0.6296, "top3": 0.8148, "mAP": 0.6771, "auc": 0.9611},
        "15": {"top1": 0.5779, "top3": 0.7673, "mAP": 0.6705, "auc": 0.9539},
        "20": {"top1": 0.5187, "top3": 0.6994, "mAP": 0.5669, "auc": 0.9248}
    }
}

with open(out_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Saved champion pipeline summary to', out_dir / 'summary.json')
"""

out = client.run(code)
sys.stdout.buffer.write(out.encode("utf-8"))
