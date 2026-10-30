import sys
from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0
)

code = r"""
import os, subprocess, json
from pathlib import Path

print("GPU:")
try:
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"]).decode())
except Exception as e:
    print(e)

print("Files in /marimo:")
for p in Path("/marimo").iterdir():
    print(p, "(dir)" if p.is_dir() else f"({p.stat().st_size} bytes)")

print("\nFiles in /marimo/pretrained:")
if Path("/marimo/pretrained").exists():
    for p in Path("/marimo/pretrained").iterdir():
        print(p, f"({p.stat().st_size / 1e6:.2f} MB)")

print("\nFiles in /marimo/dataset/cache:")
if Path("/marimo/dataset/cache").exists():
    for p in Path("/marimo/dataset/cache").iterdir():
        print(p, f"({p.stat().st_size / 1e6:.2f} MB)")
"""

res = c.run(code)
print(res)
