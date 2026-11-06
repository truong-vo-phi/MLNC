import base64
import json
from pathlib import Path
import requests

URL = "https://sb-3632b8f8da56ce54.sb.molab.run"
TOKEN = "1c3cd423d853201370f94f61624c8bd1cca4002a61f133c8178c7fbcfb4509b1"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

def get_session():
    r = requests.get(f"{URL}/api/sessions", headers=headers, timeout=15)
    r.raise_for_status()
    s_key = list(r.json().keys())[0]
    return s_key

def execute_code(code: str, session_id: str):
    h = {**headers, "Marimo-Session-Id": session_id}
    resp = requests.post(
        f"{URL}/api/kernel/execute",
        headers=h,
        json={"code": code},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.status_code

session_id = get_session()
print(f"Connected to session: {session_id}")

# 1. Reset remote file
execute_code("""
from pathlib import Path
p = Path('/marimo/review-3.zip')
if p.exists(): p.unlink()
p.parent.mkdir(parents=True, exist_ok=True)
print('Initialized /marimo/review-3.zip')
""", session_id)

# 2. Upload zip in chunks
zip_bytes = Path("scratch/review3_deploy.zip").read_bytes()
b64_str = base64.b64encode(zip_bytes).decode("ascii")
chunk_size = 50000
total_len = len(b64_str)
num_chunks = (total_len + chunk_size - 1) // chunk_size
print(f"Sending zip: {len(zip_bytes)/1024:.1f} KB in {num_chunks} chunks...")

for idx in range(num_chunks):
    chunk = b64_str[idx * chunk_size : (idx + 1) * chunk_size]
    code = f"""
import base64
with open('/marimo/review-3.zip', 'ab') as f:
    f.write(base64.b64decode({json.dumps(chunk)}))
"""
    execute_code(code, session_id)
    print(f"  Chunk {idx+1}/{num_chunks} sent")

# 3. Unzip on remote
unpack_code = """
import zipfile, shutil
from pathlib import Path

target = Path('/marimo/review-3')
if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile('/marimo/review-3.zip') as z:
    z.extractall(target)

print('Successfully extracted to /marimo/review-3')
print('Extracted files:', len(list(target.rglob('*'))))
"""
execute_code(unpack_code, session_id)
print("Unpack command sent successfully!")
