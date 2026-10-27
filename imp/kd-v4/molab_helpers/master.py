import os, shutil, subprocess, sys, time
from pathlib import Path
out = Path("/marimo/output")
status = out / "master.status"
def stamp(message):
    status.write_text(message)
    print(time.strftime("%H:%M:%S"), message, flush=True)
def step(name, script, log_name):
    stamp(f"{name}: dang chay")
    started = time.time()
    with open(out / log_name, "ab") as handle:
        code = subprocess.run([sys.executable, "-u", str(out / script)], cwd="/marimo", stdout=handle, stderr=subprocess.STDOUT).returncode
    if code != 0:
        stamp(f"{name}: LOI (ma thoat {code}), dung")
        sys.exit(code)
    print(f"  {name} xong sau {time.time() - started:.0f}s", flush=True)
step("download", "download.py", "download.log")
step("cache", "cache.py", "cache.log")
shutil.rmtree("/marimo/output/resume/e2e-final-test", ignore_errors=True)
step("e2e", "e2e.py", "e2e.log")
step("train", "train.py", "train.log")
stamp("DONE")
