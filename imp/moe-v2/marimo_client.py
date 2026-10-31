"""
Direct Python Client for Marimo Remote Code Execution.
Bypasses Windows bash / jq command-line argument limits.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Iterator

MARIMO_URL = "https://sb-f69e127a8595d1c6.sb.molab.run"
MARIMO_TOKEN = "cfd8477208df21766d10051c990c71a18592ffb8a69611e636531aa3782cc8ed"


def get_session_id(url: str = MARIMO_URL, token: str = MARIMO_TOKEN) -> str:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/sessions",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "curl/8.4.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    sessions = list(data.keys())
    if not sessions:
        raise RuntimeError("No active marimo sessions found on server.")
    return sessions[0]


def execute_remote(code: str, url: str = MARIMO_URL, token: str = MARIMO_TOKEN) -> int:
    session_id = get_session_id(url, token)
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/kernel/execute",
        headers={
            "Authorization": f"Bearer {token}",
            "Marimo-Session-Id": session_id,
            "Content-Type": "application/json",
            "User-Agent": "curl/8.4.0",
        },
        data=json.dumps({"code": code}).encode("utf-8"),
    )

    exit_code = 0
    current_event = ""

    with urllib.request.urlopen(req, timeout=3600) as resp:
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
                    # success or error
                    success = payload.get("success", True)
                    if not success:
                        exit_code = 1
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            code_to_run = f.read()
    else:
        code_to_run = sys.stdin.read()

    rc = execute_remote(code_to_run)
    sys.exit(rc)
