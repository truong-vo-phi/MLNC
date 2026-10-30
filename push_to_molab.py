import base64
import json
import time
import requests
from pathlib import Path

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# 1. Get session ID
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
    print(f"Sending {local_path.name} ({len(data) / 1e6:.2f} MB, {total_len} b64 chars)...")

    # Clear destination file
    clear_code = f"""
from pathlib import Path
p = Path('{remote_path}')
p.parent.mkdir(parents=True, exist_ok=True)
if p.exists(): p.unlink()
"""
    run_code_on_molab(clear_code)

    # Send in chunks
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
        print(f"  Sent chunk {i // chunk_size + 1} / {(total_len + chunk_size - 1) // chunk_size}")

    verify_code = f"""
from pathlib import Path
p = Path('{remote_path}')
print(f"Received {remote_path}: {{p.stat().st_size / 1e6:.2f}} MB")
"""
    run_code_on_molab(verify_code)


if __name__ == "__main__":
    pkg_zip = Path(r"D:\PROJECT\MLNC\package_to_molab.zip")
    manifest_zip = Path(r"D:\PROJECT\MLNC\manifests.zip")

    # 1. Send package_to_molab.zip
    send_file(pkg_zip, "/marimo/package_to_molab.zip")

    # 2. Extract package_to_molab.zip
    extract_pkg_code = """
import zipfile
from pathlib import Path

z_path = Path('/marimo/package_to_molab.zip')
with zipfile.ZipFile(z_path, 'r') as z:
    z.extractall('/marimo')
print("Extracted package_to_molab.zip to /marimo successfully!")
"""
    run_code_on_molab(extract_pkg_code)

    # 3. Send manifests.zip
    send_file(manifest_zip, "/marimo/dataset/manifests.zip")

    # 4. Extract manifests.zip
    extract_manifest_code = """
import zipfile
from pathlib import Path

z_path = Path('/marimo/dataset/manifests.zip')
for dest in [Path('/marimo/dataset'), Path('/marimo/dataset/mix-dataset-v2')]:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z_path, 'r') as z:
        z.extractall(dest)
    print(f"Extracted manifests.zip to {dest} successfully!")
"""
    run_code_on_molab(extract_manifest_code)
