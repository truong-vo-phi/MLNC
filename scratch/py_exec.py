import urllib.request
import json
import sys

BASE = "https://sb-f485279477cfa9a2.sb.molab.run"
TOKEN = "c69e05b6193257d19c3fbd42a439db8b293b85d6efcba77a4a3ec04d6e8640a5"

def get_session_id():
    req = urllib.request.Request(
        f"{BASE}/api/sessions",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "User-Agent": "curl/8.7.1",
        }
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        keys = list(data.keys())
        if not keys:
            raise RuntimeError("No active session")
        return keys[0]

def run_remote_code(code_str: str):
    session_id = get_session_id()
    data = json.dumps({"code": code_str}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/kernel/execute",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "Marimo-Session-Id": session_id,
            "User-Agent": "curl/8.7.1",
        }
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8", errors="replace").strip()
            if line_str.startswith("data:"):
                payload_str = line_str[5:].strip()
                try:
                    payload = json.loads(payload_str)
                    if "data" in payload:
                        sys.stdout.write(payload["data"])
                        sys.stdout.flush()
                except Exception:
                    pass

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "-c":
        code = sys.argv[2]
    elif len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            code = f.read()
    else:
        code = 'print("PYTHON_CLIENT_ACTIVE_OK")'
    run_remote_code(code)
