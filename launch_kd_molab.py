import subprocess
import sys
from pathlib import Path
from scripts.molab_deploy import MolabClient

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

client = MolabClient(url=URL, token=TOKEN, timeout=120.0)

code = r"""
import os, subprocess, time
from pathlib import Path

# Kill any existing main.py processes
subprocess.run(['pkill', '-9', '-f', 'main.py'], capture_output=True)
time.sleep(1)

out_dir = Path('/marimo/output/kd-v5')
out_dir.mkdir(parents=True, exist_ok=True)
log_file = out_dir / 'train.log'
if log_file.exists():
    log_file.unlink()

cmd = [
    'python', '-u', '/marimo/imp/kd-v5/main.py', 'train',
    '--noise-cache', '/marimo/dataset/cache',
    '--output-dir', '/marimo/output/kd-v5',
    '--device', 'cuda'
]

env = {
    **os.environ,
    'PYTHONUNBUFFERED': '1',
    'MLNC_MIXTURE_36L': '/marimo/dataset/mix-dataset-v2'
}

with open(log_file, 'w', encoding='utf-8') as f:
    proc = subprocess.Popen(
        cmd,
        cwd='/marimo/imp/kd-v5',
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env
    )

print(f'KD training launched with PID {proc.pid}! Logging to {log_file}')
"""

res = client.run(code)
print(res)
