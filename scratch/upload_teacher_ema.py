from __future__ import annotations

import io
import json
import os
import sys
import time
import torch
from pathlib import Path
from scripts.molab_deploy import MolabClient, write_remote

def main() -> None:
    p = Path(r"D:\PROJECT\MLNC\pretrained\beat-ft\beats_teacher_ema_standalone.pt")
    if not p.exists():
        print(f"Error: {p} does not exist")
        return

    print("Loading local checkpoint...")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    sd_fp16 = {k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v for k, v in sd.items()}
    
    buf = io.BytesIO()
    torch.save({"model": sd_fp16, "config": ckpt.get("config")}, buf)
    data = buf.getvalue()
    total_bytes = len(data)
    print(f"Serialized fp16 size: {total_bytes / (1024 * 1024):.2f} MB")

    client = MolabClient(
        url="https://sb-d3f93608a9dca958.sb.molab.run",
        token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
        timeout=60.0,
    )

    chunk_size = 4 * 1024 * 1024  # 4 MB per chunk
    parts: list[str] = []
    t0 = time.time()
    
    # Clean remote temp parts dir
    client.run("import shutil, os; shutil.rmtree('/marimo/tmp_upload', ignore_errors=True); os.makedirs('/marimo/tmp_upload', exist_ok=True)")

    num_chunks = (total_bytes + chunk_size - 1) // chunk_size
    print(f"Uploading {num_chunks} chunks (4MB each)...")

    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        part_name = f"tmp_upload/part_{i:04d}.bin"
        write_remote(client, {part_name: chunk}, force=True)
        if (i + 1) % 5 == 0 or i == num_chunks - 1:
            print(f"  Chunk {i + 1}/{num_chunks} uploaded ({((i + 1) / num_chunks) * 100:.1f}%)")

    elapsed = time.time() - t0
    print(f"All chunks uploaded in {elapsed:.2f}s! Assembling on Molab...")

    # Assemble on Molab
    assemble_code = f"""
import os, glob, shutil, torch

parts = sorted(glob.glob('/marimo/tmp_upload/part_*.bin'))
dest = '/marimo/pretrained/beats_teacher_ema_standalone.pt'
os.makedirs('/marimo/pretrained', exist_ok=True)

with open(dest, 'wb') as out_f:
    for p in parts:
        with open(p, 'rb') as in_f:
            out_f.write(in_f.read())

shutil.rmtree('/marimo/tmp_upload', ignore_errors=True)
sz = os.path.getsize(dest)
print(f'Successfully assembled {dest}, size: {sz / (1024*1024):.2f} MB')

ckpt = torch.load(dest, map_location='cpu', weights_only=False)
print(f'Keys: {list(ckpt.keys())}, model weights count: {len(ckpt["model"])}')
"""
    print(client.run(assemble_code))

if __name__ == "__main__":
    main()
