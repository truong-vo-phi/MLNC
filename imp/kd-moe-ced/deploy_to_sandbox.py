import base64
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

LOCAL_DIR = Path(__file__).resolve().parent
MARIMO_URL = os.environ.get("MARIMO_URL", "https://sb-c04297299c79cc0d.sb.molab.run")
MARIMO_TOKEN = os.environ.get("MARIMO_TOKEN", "95900018fcf7379c970f9596ce7758087d253b3c7d0cca8040f51f30edc0729e")

def get_session_id():
    req = urllib.request.Request(
        f"{MARIMO_URL.rstrip('/')}/api/sessions",
        headers={
            "Authorization": f"Bearer {MARIMO_TOKEN}",
            "User-Agent": "curl/8.4.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    sessions = list(data.keys())
    if not sessions:
        raise RuntimeError("No active marimo sessions found on server.")
    return sessions[0]

def execute_remote(code: str):
    session_id = get_session_id()
    req = urllib.request.Request(
        f"{MARIMO_URL.rstrip('/')}/api/kernel/execute",
        headers={
            "Authorization": f"Bearer {MARIMO_TOKEN}",
            "Marimo-Session-Id": session_id,
            "Content-Type": "application/json",
            "User-Agent": "curl/8.4.0",
        },
        data=json.dumps({"code": code}).encode("utf-8"),
    )
    exit_code = 0
    current_event = ""
    with urllib.request.urlopen(req, timeout=180) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                current_event = line[len("event: ") :].strip()
            elif line.startswith("data:"):
                payload_str = line[len("data: ") :]
                try:
                    payload = json.loads(payload_str)
                except Exception:
                    continue

                if current_event == "stdout":
                    sys.stdout.buffer.write(payload.get("data", "").encode("utf-8", errors="replace"))
                    sys.stdout.buffer.flush()
                elif current_event == "stderr":
                    sys.stderr.buffer.write(payload.get("data", "").encode("utf-8", errors="replace"))
                    sys.stderr.buffer.flush()
                elif current_event == "done":
                    success = payload.get("success", True)
                    if not success:
                        exit_code = 1
    return exit_code

def main():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(LOCAL_DIR):
            rel_root = os.path.relpath(root, LOCAL_DIR)
            if any(p in rel_root for p in [".git", "__pycache__", ".pytest_cache"]):
                continue
            for f in files:
                if f.endswith((".b64", ".zip", "deploy_to_sandbox.py")):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, LOCAL_DIR)
                z.write(full_path, rel_path)

    b64_data = base64.b64encode(bio.getvalue()).decode("utf-8")
    print(f"Compressed package size in b64: {len(b64_data)} characters")

    remote_code = f"""
import base64
import io
import os
import shutil
import zipfile

b64_data = '''{b64_data}'''
dest_dir = '/marimo/imp/kd-moe-ced'
os.makedirs(dest_dir, exist_ok=True)

bio = io.BytesIO(base64.b64decode(b64_data))
with zipfile.ZipFile(bio, 'r') as z:
    z.extractall(dest_dir)

# Also copy src and config directly to /marimo
if os.path.exists('/marimo/src'):
    shutil.rmtree('/marimo/src')
shutil.copytree('/marimo/imp/kd-moe-ced/src', '/marimo/src')

if os.path.exists('/marimo/config'):
    shutil.rmtree('/marimo/config')
shutil.copytree('/marimo/imp/kd-moe-ced/config', '/marimo/config')

print('Successfully deployed kd-moe-ced to /marimo/imp/kd-moe-ced and /marimo/src!')
print('Contents of /marimo/src:', os.listdir('/marimo/src'))
print('Contents of /marimo/src/models:', os.listdir('/marimo/src/models'))
"""
    rc = execute_remote(remote_code)
    print("Remote execution status code:", rc)

if __name__ == "__main__":
    main()
