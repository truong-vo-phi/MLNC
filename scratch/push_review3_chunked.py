import base64
from pathlib import Path
import subprocess
import time

zip_bytes = Path("scratch/review3_deploy.zip").read_bytes()
b64_str = base64.b64encode(zip_bytes).decode("ascii")

chunk_size = 20000
total_len = len(b64_str)
num_chunks = (total_len + chunk_size - 1) // chunk_size
print(f"Total b64 length: {total_len} bytes across {num_chunks} chunks.")

# 1. Reset remote file
init_cmd = """from pathlib import Path
p = Path('/marimo/review-3.zip')
if p.exists(): p.unlink()
p.parent.mkdir(parents=True, exist_ok=True)
print('Initialized /marimo/review-3.zip')
"""

def run_remote(code_snippet):
    cmd = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"d:/PROJECT/MLNC/.agents/skills/marimo-pair/scripts/execute-code.sh",
        "--url", "https://sb-3632b8f8da56ce54.sb.molab.run/",
        "--token", "1c3cd423d853201370f94f61624c8bd1cca4002a61f133c8178c7fbcfb4509b1",
        "-c", code_snippet,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed: {res.stderr}\n{res.stdout}")
    return res.stdout

run_remote(init_cmd)

# 2. Append chunks
for idx in range(num_chunks):
    chunk = b64_str[idx * chunk_size : (idx + 1) * chunk_size]
    append_snippet = f"""import base64
with open('/marimo/review-3.zip', 'ab') as f:
    f.write(base64.b64decode('{chunk}'))
"""
    run_remote(append_snippet)
    print(f"  Sent chunk {idx + 1}/{num_chunks}")

# 3. Unzip on remote
unzip_cmd = """import zipfile, shutil
from pathlib import Path

target = Path('/marimo/review-3')
if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile('/marimo/review-3.zip') as z:
    z.extractall(target)

print('Unpacked successfully to', target)
print('Total files extracted:', len(list(target.rglob('*'))))
"""
out = run_remote(unzip_cmd)
print("Unpack output:", out)
