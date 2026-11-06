import base64
import io
import json
import zipfile
from pathlib import Path
import requests

URL = "https://sb-83326b2d8a8049e4.sb.molab.run"
TOKEN = "a6711c9c0fc6b1d77c8fdf2b6e9c985802f2e7c36d66984f54fd0382e1e612a1"

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
            try:
                payload = json.loads(line[6:])
                if "data" in payload and isinstance(payload["data"], str):
                    print(payload["data"], end="")
            except Exception:
                pass


# 2. Create in-memory zip of imp/review-3/src
src_dir = Path(r"d:\PROJECT\MLNC\imp\review-3\src")
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in src_dir.rglob("*.py"):
        rel_path = file_path.relative_to(src_dir.parent)
        zf.write(file_path, arcname=str(rel_path).replace("\\", "/"))

zip_bytes = zip_buffer.getvalue()
print(f"Created zip archive: {len(zip_bytes) / 1024:.2f} KB")

# 3. Transfer zip bytes via base64
b64_str = base64.b64encode(zip_bytes).decode("ascii")

send_code = f"""
import base64
import io
import zipfile
from pathlib import Path

b64_data = "{b64_str}"
zip_bytes = base64.b64decode(b64_data)
with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
    zf.extractall("/marimo")
print("Extracted src/ to /marimo:")
print([str(p.relative_to('/marimo')) for p in Path('/marimo/src').rglob('*.py') if 'ced_moe_split' in str(p)])
"""

print("Extracting on molab...")
run_code_on_molab(send_code)
