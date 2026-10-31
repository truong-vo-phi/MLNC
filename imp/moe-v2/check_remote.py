import sys
from pathlib import Path

# Add current dir
sys.path.insert(0, str(Path(__file__).parent))
import marimo_client

code = """
import os
import subprocess

print("=== DISK SPACE ===")
print(subprocess.check_output("df -h", shell=True).decode())

print("=== FIND FILES > 50MB ===")
out = subprocess.check_output("find /marimo -size +50M 2>/dev/null", shell=True).decode()
print(out)

print("=== CHECK /tmp and /root ===")
for path in ["/tmp", "/root", "/marimo"]:
    if os.path.exists(path):
        print(f"Contents of {path}:", os.listdir(path)[:20])
"""

marimo_client.execute_remote(code)
