import base64
import json
import os
from pathlib import Path
import sys
import urllib.request

LOCAL_DIR = Path(__file__).resolve().parent
MARIMO_URL = os.environ.get("MARIMO_URL", "https://sb-1f2f333bbf05f61f.sb.molab.run")
MARIMO_TOKEN = os.environ.get("MARIMO_TOKEN", "ffca46ee91177bf963ed3e88946f934b2a8b15a63fff65c41fd4a806d447844d")

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

def execute_capture(code: str, timeout: int = 120) -> tuple[str, str]:
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
    stdout_chunks = []
    stderr_chunks = []
    current_event = ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                current_event = line[len("event: ") :].strip()
            elif line.startswith("data:"):
                try:
                    payload = json.loads(line[len("data: ") :])
                    if current_event == "stdout":
                        stdout_chunks.append(payload.get("data", ""))
                    elif current_event == "stderr":
                        stderr_chunks.append(payload.get("data", ""))
                except Exception:
                    continue
    return "".join(stdout_chunks), "".join(stderr_chunks)

def download_file(remote_path: str, local_path: Path, chunk_size_mb: int = 2):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    code_size = f"""
import os
p = '{remote_path}'
print('FILE_SIZE=' + str(os.path.getsize(p) if os.path.exists(p) else -1))
"""
    stdout, stderr = execute_capture(code_size)
    file_size = -1
    for line in stdout.splitlines():
        if line.startswith("FILE_SIZE="):
            file_size = int(line.split("=")[1].strip())

    if file_size <= 0:
        print(f"Error: Remote file {remote_path} not found or empty (size: {file_size}). Stderr: {stderr}")
        return False

    print(f"Downloading {remote_path} ({file_size / (1024*1024):.2f} MB) -> {local_path}...")
    chunk_bytes = chunk_size_mb * 1024 * 1024
    total_chunks = (file_size + chunk_bytes - 1) // chunk_bytes

    start_tag = "===CHUNK_START==="
    end_tag = "===CHUNK_END==="

    with open(local_path, "wb") as f_out:
        for chunk_idx in range(total_chunks):
            offset = chunk_idx * chunk_bytes
            length = min(chunk_bytes, file_size - offset)
            code_chunk = f"""
import base64
import sys
with open('{remote_path}', 'rb') as f:
    f.seek({offset})
    data = f.read({length})
b64 = base64.b64encode(data).decode('utf-8')
sys.stdout.write('{start_tag}' + b64 + '{end_tag}')
sys.stdout.flush()
"""
            chunk_out, err = execute_capture(code_chunk, timeout=120)
            if start_tag not in chunk_out or end_tag not in chunk_out:
                print(f"Error on chunk {chunk_idx + 1}: tags not found in response. Err: {err}")
                return False

            b64_str = chunk_out.split(start_tag)[1].split(end_tag)[0].strip()
            raw = base64.b64decode(b64_str)
            f_out.write(raw)
            print(f"  Chunk {chunk_idx + 1}/{total_chunks} ({len(raw)/(1024*1024):.2f} MB) saved.")

    print(f"Successfully saved {local_path} ({os.path.getsize(local_path)} bytes)!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 2:
        remote = sys.argv[1]
        local = Path(sys.argv[2])
    else:
        remote = "/marimo/result.csv"
        local = LOCAL_DIR / "result.csv"
    download_file(remote, local)
