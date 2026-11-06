import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
lines = Path("imp/kd-moe-ced/notebook.py").read_text(encoding="utf-8").splitlines()

print("=== LINES 500-580 ===")
for j in range(500, 580):
    if j < len(lines):
        print(f"{j+1}: {lines[j]}")
