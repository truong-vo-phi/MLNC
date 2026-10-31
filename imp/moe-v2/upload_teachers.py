from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

# Add imp/moe-v2 to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "imp" / "moe-v2"))
import marimo_client

def upload_file_chunked(
    local_path: Path,
    remote_path: str,
    chunk_size_bytes: int = 4 * 1024 * 1024,
    as_fp16: bool = False,
) -> None:
    print(f"\n--- Uploading {local_path.name} to {remote_path} ---")
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Prepare data
    if as_fp16:
        print("Loading and converting to fp16...")
        import torch
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            sd = ckpt["model"]
            sd_fp16 = {k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v for k, v in sd.items()}
            ckpt["model"] = sd_fp16
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            sd_fp16 = {k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v for k, v in sd.items()}
            ckpt["state_dict"] = sd_fp16
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        data = buf.getvalue()
        del ckpt, sd, sd_fp16
    else:
        data = local_path.read_bytes()

    total_len = len(data)
    total_mb = total_len / 1e6
    num_chunks = (total_len + chunk_size_bytes - 1) // chunk_size_bytes
    print(f"Total size: {total_mb:.2f} MB across {num_chunks} chunks.")

    # Prepare remote dir and remove existing file
    init_code = f"""
from pathlib import Path
p = Path('{remote_path}')
p.parent.mkdir(parents=True, exist_ok=True)
if p.exists():
    p.unlink()
"""
    marimo_client.execute_remote(init_code)

    t0 = time.time()
    for i in range(num_chunks):
        chunk = data[i * chunk_size_bytes : (i + 1) * chunk_size_bytes]
        b64 = base64.b64encode(chunk).decode("ascii")
        code = f"""
import base64
chunk_bytes = base64.b64decode('{b64}')
with open('{remote_path}', 'ab') as f:
    f.write(chunk_bytes)
"""
        marimo_client.execute_remote(code)
        elapsed = time.time() - t0
        rate = ((i + 1) * (len(chunk)) / 1e6) / max(elapsed, 0.1)
        if (i + 1) % 5 == 0 or (i + 1) == num_chunks:
            pct = ((i + 1) / num_chunks) * 100
            print(f"  Chunk {i + 1:3d}/{num_chunks} ({pct:5.1f}%) | {rate:.2f} MB/s | Elapsed: {elapsed:.1f}s")

    verify_code = f"""
import os
sz = os.path.getsize('{remote_path}') / 1e6
print(f"Verified remote file: {remote_path} -> {{sz:.2f}} MB")
"""
    marimo_client.execute_remote(verify_code)
    print(f"Done uploading {local_path.name} in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    t_start = time.time()
    # 1. Fold0 teacher
    p_fold0 = Path("pretrained/beat-ft/beats_teacher_fold0_weights.pt")
    upload_file_chunked(p_fold0, "/marimo/pretrained/beats_teacher_fold0_weights.pt", chunk_size_bytes=4*1024*1024, as_fp16=False)

    # 2. EMA teacher (convert to fp16 to save 180MB of upload!)
    p_ema = Path("pretrained/beat-ft/beats_teacher_ema_standalone.pt")
    upload_file_chunked(p_ema, "/marimo/pretrained/beats_teacher_ema_standalone.pt", chunk_size_bytes=4*1024*1024, as_fp16=True)

    print(f"\nAll teachers uploaded successfully in {time.time() - t_start:.1f}s!")
