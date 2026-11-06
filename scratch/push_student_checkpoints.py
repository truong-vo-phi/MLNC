import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.molab_deploy import MolabClient, write_remote

URL = "https://sb-991df8e09723e86f.sb.molab.run"
TOKEN = "a588d69b09c1ac17fd755b27a8f44520db8f5c1e810f0ffa92184beae0bc0b24"

client = MolabClient(url=URL, token=TOKEN, timeout=120.0)

p_stage1 = ROOT / "output/checkpoint/kdmoe-student-2026-10-11-20-52-31/stage1/last_best_checkpoint.pt"
p_stage2 = ROOT / "output/checkpoint/kdmoe-student-2026-10-11-20-52-31/stage2/last_best_checkpoint.pt"

files = {
    "output/moe-from-scratch/champion/low/last_best_checkpoint.pt": p_stage1.read_bytes(),
    "output/moe-from-scratch/champion/high/last_best_checkpoint.pt": p_stage2.read_bytes(),
}

print(f"Uploading 2 student checkpoints ({sum(len(v) for v in files.values()) / (1024**2):.2f} MB)...")
done = write_remote(client, files, force=True)
print("Uploaded:", done)
