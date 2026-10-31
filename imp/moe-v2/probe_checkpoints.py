import glob
import os

print("=== Checkpoints in /marimo ===")
for p in sorted(glob.glob("/marimo/**/*.pt", recursive=True)):
    sz_mb = os.path.getsize(p) / 1024 / 1024
    print(f"  {p} ({sz_mb:.1f} MB)")

print("\n=== Checkpoints in /marimo/checkpoints ===")
if os.path.exists("/marimo/checkpoints"):
    print(os.listdir("/marimo/checkpoints"))

print("\n=== Checkpoints in /marimo/moe-v2/checkpoints ===")
if os.path.exists("/marimo/moe-v2/checkpoints"):
    print(os.listdir("/marimo/moe-v2/checkpoints"))

print("\n=== Check other .pt or .ckpt files ===")
for p in sorted(glob.glob("/**/*.pt", recursive=False)):
    print(f"  {p}")
