import base64
import time
from pathlib import Path
from scripts.molab_deploy import MolabClient

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

client = MolabClient(url=URL, token=TOKEN, timeout=60.0)

local_path = Path("pretrained/beat-ft/beats_teacher_fold0_weights.pt")
remote_path = "/marimo/pretrained/beats_teacher_fold0_weights.pt"
chunk_size = 4_000_000  # ~3MB raw per chunk

data = local_path.read_bytes()
b64 = base64.b64encode(data).decode("ascii")
total_len = len(b64)
total_chunks = (total_len + chunk_size - 1) // chunk_size
print(f"Uploading {local_path.name}: {len(data) / 1e6:.2f} MB in {total_chunks} chunks...")

# Clear destination file
client.run(f"""
from pathlib import Path
p = Path('{remote_path}')
p.parent.mkdir(parents=True, exist_ok=True)
if p.exists(): p.unlink()
""")

t0 = time.time()
for i in range(0, total_len, chunk_size):
    chunk = b64[i : i + chunk_size]
    chunk_idx = i // chunk_size + 1
    code = f"""
import base64
from pathlib import Path
with open('{remote_path}', 'ab') as f:
    f.write(base64.b64decode('{chunk}'))
"""
    client.run(code)
    elapsed = time.time() - t0
    rate = ((chunk_idx * (chunk_size * 0.75)) / 1e6) / max(elapsed, 0.1)
    if chunk_idx % 5 == 0 or chunk_idx == total_chunks:
        print(f"  Chunk {chunk_idx}/{total_chunks} ({chunk_idx/total_chunks*100:.1f}%) - {rate:.2f} MB/s")

# Verify size
out = client.run(f"""
from pathlib import Path
p = Path('{remote_path}')
print(f"Verified remote size: {{p.stat().st_size / 1e6:.2f}} MB")
""")
print(out.strip())
