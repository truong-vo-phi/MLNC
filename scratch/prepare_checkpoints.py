import os
import sys
import base64
from pathlib import Path

stage1_path = Path("output/checkpoint/kdmoe-student-2026-10-11-20-52-31/stage1/last_best_checkpoint.pt")
stage2_path = Path("output/checkpoint/kdmoe-student-2026-10-11-20-52-31/stage2/last_best_checkpoint.pt")

print("Stage1 exists:", stage1_path.exists(), stage1_path.stat().st_size)
print("Stage2 exists:", stage2_path.exists(), stage2_path.stat().st_size)

b1 = base64.b64encode(stage1_path.read_bytes()).decode("ascii")
b2 = base64.b64encode(stage2_path.read_bytes()).decode("ascii")

with open("scratch/stage1_b64.txt", "w") as f:
    f.write(b1)
with open("scratch/stage2_b64.txt", "w") as f:
    f.write(b2)

print("Saved b64 files. Stage1 len:", len(b1), "Stage2 len:", len(b2))
