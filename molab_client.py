import base64
import json
import sys
import requests
from pathlib import Path

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

def get_session():
    r = requests.get(f"{URL}/api/sessions", headers=headers, timeout=15)
    r.raise_for_status()
    s_key = list(r.json().keys())[0]
    return s_key

def execute_code(code: str, timeout: int = 120):
    s_key = get_session()
    h = {**headers, "Marimo-Session-Id": s_key}
    resp = requests.post(
        f"{URL}/api/kernel/execute",
        headers=h,
        json={"code": code},
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    out = []
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            payload = json.loads(line[6:])
            if "data" in payload and isinstance(payload["data"], str):
                sys.stdout.write(payload["data"])
                sys.stdout.flush()
                out.append(payload["data"])
    return "".join(out)

def send_file(local_path: Path, remote_path: str, chunk_size: int = 1_500_000):
    data = local_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    total_len = len(b64)
    print(f"Sending {local_path.name} ({len(data)/1e6:.2f} MB)...")

    clear_code = f"""
from pathlib import Path
p = Path('{remote_path}')
p.parent.mkdir(parents=True, exist_ok=True)
if p.exists(): p.unlink()
"""
    execute_code(clear_code)

    for i in range(0, total_len, chunk_size):
        chunk = b64[i : i + chunk_size]
        append_code = f"""
import base64
from pathlib import Path
p = Path('{remote_path}')
with open(p, 'ab') as f:
    f.write(base64.b64decode('{chunk}'))
"""
        execute_code(append_code)
        print(f"  chunk {i // chunk_size + 1} / {(total_len + chunk_size - 1) // chunk_size}")

    verify_code = f"""
from pathlib import Path
p = Path('{remote_path}')
print(f"Received {remote_path}: {{p.stat().st_size / 1e6:.2f}} MB")
"""
    execute_code(verify_code)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        code_to_run = sys.stdin.read()
        execute_code(code_to_run)
