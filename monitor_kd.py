import sys
from scripts.molab_deploy import MolabClient

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

client = MolabClient(url=URL, token=TOKEN, timeout=30.0)

sys.stdout.reconfigure(encoding='utf-8')

code = r"""
import os, subprocess
from pathlib import Path

res = subprocess.run(['pgrep', '-f', 'main.py'], capture_output=True, text=True)
pids = res.stdout.strip().split()
running = len(pids) > 0

log_path = Path('/marimo/output/kd-v5/train.log')
if log_path.exists():
    text = log_path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    clean_lines = [l for l in lines if 'INFO' in l or ('epoch' in l and 'batch/s' not in l)]
    print(f'Total log lines: {len(lines)} | Status: {"RUNNING (PIDs: " + ", ".join(pids) + ")" if running else "FINISHED"}')
    print('--- Milestone / Epoch logs ---')
    for l in clean_lines[-25:]:
        print(l)
    print('--- Latest 3 lines ---')
    for l in lines[-3:]:
        print(l)
else:
    print('Log file not found yet.')
"""

res = client.run(code)
print(res)
