import base64
import json
import os
from pathlib import Path
import sys
import time
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

def execute_capture(code: str, session_id: str, timeout: int = 45) -> tuple[str, str]:
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

def download_file_fast(remote_path: str, local_path: Path, chunk_kb: int = 40, max_retries: int = 3):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    session_id = get_session_id()

    code_size = f"""
import os
p = '{remote_path}'
print('FILE_SIZE=' + str(os.path.getsize(p) if os.path.exists(p) else -1))
"""
    stdout, stderr = execute_capture(code_size, session_id)
    file_size = -1
    for line in stdout.splitlines():
        if line.startswith("FILE_SIZE="):
            file_size = int(line.split("=")[1].strip())

    if file_size <= 0:
        print(f"Error: Remote file {remote_path} not found or empty (size: {file_size}). Stderr: {stderr}")
        return False

    chunk_bytes = chunk_kb * 1024
    total_chunks = (file_size + chunk_bytes - 1) // chunk_bytes

    # Check existing partial file for resuming
    start_chunk = 0
    if local_path.is_file():
        existing_size = os.path.getsize(local_path)
        if existing_size == file_size:
            print(f"File {local_path} already fully downloaded ({file_size} bytes).")
            return True
        start_chunk = existing_size // chunk_bytes
        # Truncate to exact chunk boundary
        with open(local_path, "r+b") as f:
            f.truncate(start_chunk * chunk_bytes)
        print(f"Resuming download from chunk {start_chunk}/{total_chunks} ({start_chunk * chunk_bytes / (1024*1024):.2f} MB)...")
    else:
        print(f"Downloading {remote_path} ({file_size / (1024*1024):.2f} MB, {total_chunks} chunks of {chunk_kb}KB) -> {local_path}...")

    t0 = time.time()
    start_tag = "[[B64_START]]"
    end_tag = "[[B64_END]]"

    with open(local_path, "ab") as f_out:
        for chunk_idx in range(start_chunk, total_chunks):
            offset = chunk_idx * chunk_bytes
            length = min(chunk_bytes, file_size - offset)
            code_chunk = f"""
import base64
with open('{remote_path}', 'rb') as f:
    f.seek({offset})
    data = f.read({length})
print('{start_tag}' + base64.b64encode(data).decode('ascii') + '{end_tag}')
"""
            success = False
            for attempt in range(max_retries):
                try:
                    chunk_out, err = execute_capture(code_chunk, session_id, timeout=45)
                    if start_tag in chunk_out and end_tag in chunk_out:
                        b64_str = chunk_out.split(start_tag)[1].split(end_tag)[0].strip()
                        raw = base64.b64decode(b64_str)
                        f_out.write(raw)
                        f_out.flush()
                        success = True
                        break
                    else:
                        time.sleep(1.0)
                except Exception as e:
                    time.sleep(1.5)

            if not success:
                print(f"Failed to fetch chunk {chunk_idx + 1} after {max_retries} attempts.")
                return False

            if (chunk_idx + 1) % 20 == 0 or (chunk_idx + 1) == total_chunks:
                pct = ((chunk_idx + 1) / total_chunks) * 100
                speed = (chunk_idx + 1) * chunk_bytes / max(1.0, (time.time() - t0)) / 1024
                print(f"  Progress: {chunk_idx + 1}/{total_chunks} ({pct:.1f}%) | Speed: {speed:.1f} KB/s")

    saved_size = os.path.getsize(local_path)
    assert saved_size == file_size, f"Size mismatch! Remote: {file_size}, Local: {saved_size}"
    print(f"Successfully downloaded {local_path} ({saved_size / (1024*1024):.2f} MB)!")
    return True

if __name__ == "__main__":
    remote = sys.argv[1] if len(sys.argv) > 1 else "/marimo/checkpoints/best_model_fp16.pt"
    local = Path(sys.argv[2]) if len(sys.argv) > 2 else LOCAL_DIR / "checkpoints" / "best_model.pt"
    download_file_fast(remote, local)
