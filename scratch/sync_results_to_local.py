import json, os, pathlib

LOCAL_RESULTS_DIR = pathlib.Path("d:/PROJECT/MLNC/experiments/results")
LOCAL_CONFIGS_DIR = pathlib.Path("d:/PROJECT/MLNC/experiments/configs")
LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

remote_files = [
    # Configs
    ("/marimo/moe/config/kd_ced_tiny.yaml", LOCAL_CONFIGS_DIR / "kd_ced_tiny_run1.yaml"),
    ("/marimo/moe/config/kd_ced_tiny_no_rms.yaml", LOCAL_CONFIGS_DIR / "kd_ced_tiny_no_rms_run2.yaml"),
    ("/marimo/moe/config/kd_ced_tiny_split_10_15.yaml", LOCAL_CONFIGS_DIR / "kd_ced_tiny_split_10_15_run3.yaml"),
    ("/marimo/moe/config/moe.yaml", LOCAL_CONFIGS_DIR / "moe_pipeline.yaml"),
    # Results & Summaries
    ("/marimo/output/moe-from-scratch/20261105-081718-fp32-math/summary.json", LOCAL_RESULTS_DIR / "run1_with_rms_summary.json"),
    ("/marimo/output/moe-no-rms/summary.json", LOCAL_RESULTS_DIR / "run2_no_rms_summary.json"),
    ("/marimo/output/moe-split-10-15/summary.json", LOCAL_RESULTS_DIR / "run3_split_10_15_summary.json"),
    ("/marimo/output/comparison_rms_vs_no_rms.json", LOCAL_RESULTS_DIR / "comparison_rms_vs_no_rms.json"),
    ("/marimo/output/run3_scores.json", LOCAL_RESULTS_DIR / "run3_scores.json"),
]

# Write a remote script that packages these files as JSON
fetch_code = f"""import json, pathlib
files = {repr([r[0] for r in remote_files])}
out = {{}}
for f in files:
    p = pathlib.Path(f)
    if p.is_file():
        out[f] = p.read_text(encoding='utf-8')
print("---PAYLOAD_START---")
print(json.dumps(out))
print("---PAYLOAD_END---")
"""

import urllib.request
BASE = "https://sb-f485279477cfa9a2.sb.molab.run"
TOKEN = "c69e05b6193257d19c3fbd42a439db8b293b85d6efcba77a4a3ec04d6e8640a5"

def fetch():
    req_sess = urllib.request.Request(f"{BASE}/api/sessions", headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(req_sess) as r:
        sess = list(json.loads(r.read().decode()).keys())[0]

    req_exec = urllib.request.Request(
        f"{BASE}/api/kernel/execute",
        data=json.dumps({"code": fetch_code}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}", "Marimo-Session-Id": sess, "User-Agent": "curl/8.7.1"}
    )
    buffer = []
    with urllib.request.urlopen(req_exec) as resp:
        for line in resp:
            line_str = line.decode("utf-8", errors="replace").strip()
            if line_str.startswith("data:"):
                try:
                    payload = json.loads(line_str[5:].strip())
                    if "data" in payload:
                        buffer.append(payload["data"])
                except Exception:
                    pass
    raw = "".join(buffer)
    if "---PAYLOAD_START---" in raw and "---PAYLOAD_END---" in raw:
        payload_json = raw.split("---PAYLOAD_START---")[1].split("---PAYLOAD_END---")[0].strip()
        data_dict = json.loads(payload_json)
        for rem, loc in remote_files:
            if rem in data_dict:
                loc.write_text(data_dict[rem], encoding="utf-8")
                print(f"[SYNCED] {rem} -> {loc} ({len(data_dict[rem])} bytes)")
            else:
                print(f"[MISSING] {rem}")
    else:
        print("Failed to parse payload. Raw:", raw[:300])

if __name__ == "__main__":
    fetch()
