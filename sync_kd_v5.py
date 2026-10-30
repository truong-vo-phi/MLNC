import base64
import json
import requests
from pathlib import Path

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

sessions_resp = requests.get(f"{URL}/api/sessions", headers=headers, timeout=10)
sessions_resp.raise_for_status()
session_id = list(sessions_resp.json().keys())[0]
print(f"Target session: {session_id}")
headers["Marimo-Session-Id"] = session_id

def run_code_on_molab(code: str) -> None:
    resp = requests.post(
        f"{URL}/api/kernel/execute",
        headers=headers,
        json={"code": code},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            payload = json.loads(line[6:])
            if "data" in payload and isinstance(payload["data"], str):
                print(payload["data"], end="")

def send_file(local_path: Path, remote_path: str, chunk_size: int = 1_500_000) -> None:
    data = local_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    total_len = len(b64)
    print(f"Sending {local_path.name} to {remote_path} ({len(data)} bytes)...")

    clear_code = f"""
from pathlib import Path
p = Path('{remote_path}')
p.parent.mkdir(parents=True, exist_ok=True)
if p.exists(): p.unlink()
"""
    run_code_on_molab(clear_code)

    for i in range(0, total_len, chunk_size):
        chunk = b64[i : i + chunk_size]
        append_code = f"""
import base64
from pathlib import Path
p = Path('{remote_path}')
with open(p, 'ab') as f:
    f.write(base64.b64decode('{chunk}'))
"""
        run_code_on_molab(append_code)

    verify_code = f"""
from pathlib import Path
p = Path('{remote_path}')
print(f"Verified {remote_path}: {{p.stat().st_size}} bytes")
"""
    run_code_on_molab(verify_code)

if __name__ == "__main__":
    files = [
        (Path("imp/kd-v5/scripts/load_dataset.py"), "/marimo/imp/kd-v5/scripts/load_dataset.py"),
        (Path("imp/kd-v5/src/train/config.py"), "/marimo/imp/kd-v5/src/train/config.py"),
        (Path("imp/kd-v5/src/train/kd_trainer.py"), "/marimo/imp/kd-v5/src/train/kd_trainer.py"),
        (Path("imp/kd-v5/config/kd_ced_tiny.yaml"), "/marimo/imp/kd-v5/config/kd_ced_tiny.yaml"),
        (Path("imp/kd-v5/main.py"), "/marimo/imp/kd-v5/main.py"),
    ]
    for local_p, remote_p in files:
        send_file(local_p, remote_p)
    print("All files synced successfully!")
