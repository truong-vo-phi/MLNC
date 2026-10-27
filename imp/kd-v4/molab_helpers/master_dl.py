import subprocess, sys, time
from pathlib import Path
out = Path("/marimo/output"); status = out / "download.status"
status.write_text("download: dang chay")
with open(out / "download.log", "ab") as handle:
    code = subprocess.run([sys.executable, "-u", str(out / "download.py")], cwd="/marimo", stdout=handle, stderr=subprocess.STDOUT).returncode
status.write_text("DONE" if code == 0 else f"LOI (ma thoat {code})")
