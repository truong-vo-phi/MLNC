import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

code = """
from pathlib import Path
for p in sorted(Path('/marimo/datasets/mix-dataset-v2').rglob('*')):
    if p.is_file():
        sz = p.stat().st_size / 1e6
        print(f'{str(p)} ({sz:.2f} MB)')
"""

execute_remote(code)
