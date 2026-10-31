import glob
import os

print("=== Checking /marimo/pretrained ===")
if os.path.exists("/marimo/pretrained"):
    print("Files in /marimo/pretrained:")
    for f in os.listdir("/marimo/pretrained"):
        sz = os.path.getsize(f"/marimo/pretrained/{f}") / 1024 / 1024
        print(f"  {f} ({sz:.1f} MB)")
else:
    print("/marimo/pretrained does not exist!")

print("\n=== Search for *beat* everywhere ===")
for root, dirs, files in os.walk("/"):
    if any(p in root for p in ["/proc", "/sys", "/dev", "/usr", "/lib", "/var", "/etc", "/root/.cache"]):
        continue
    for f in files:
        if "beat" in f.lower() or "teacher" in f.lower():
            p = os.path.join(root, f)
            print(f"Found: {p} ({os.path.getsize(p)/1024/1024:.1f} MB)")
