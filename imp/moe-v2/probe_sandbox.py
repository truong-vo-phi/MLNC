import os
import torch

print("=== Sandbox Environment Probe ===")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Capability:", torch.cuda.get_device_capability(0))
    print("Device Count:", torch.cuda.device_count())

print("\n=== Directory Listing of /marimo ===")
for root, dirs, files in os.walk("/marimo"):
    # Limit depth
    depth = root.count("/")
    if depth <= 3:
        print(f"{root} : {len(files)} files, {len(dirs)} dirs")
