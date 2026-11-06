import base64
import csv
import json
from pathlib import Path
import requests

URL = "https://sb-83326b2d8a8049e4.sb.molab.run"
TOKEN = "a6711c9c0fc6b1d77c8fdf2b6e9c985802f2e7c36d66984f54fd0382e1e612a1"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def get_headers():
    s_resp = requests.get(f"{URL}/api/sessions", headers=headers, timeout=10)
    s_resp.raise_for_status()
    session_id = list(s_resp.json().keys())[0]
    h = dict(headers)
    h["Marimo-Session-Id"] = session_id
    return h


def execute_on_molab(code: str) -> str:
    h = get_headers()
    resp = requests.post(
        f"{URL}/api/kernel/execute",
        headers=h,
        json={"code": code},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()
    out = []
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                payload = json.loads(line[6:])
                if "data" in payload and isinstance(payload["data"], str):
                    out.append(payload["data"])
            except Exception:
                pass
    return "".join(out)


def download_remote_file(remote_path: str, local_path: Path) -> bool:
    code = f"""
import base64
from pathlib import Path
p = Path('{remote_path}')
if p.is_file():
    print(base64.b64encode(p.read_bytes()).decode('ascii'))
else:
    print('__NOT_FOUND__')
"""
    b64_data = execute_on_molab(code).strip()
    if not b64_data or "__NOT_FOUND__" in b64_data:
        print(f"File not found on remote: {remote_path}")
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(base64.b64decode(b64_data))
    print(f"Downloaded: {remote_path} -> {local_path} ({local_path.stat().st_size / 1e3:.1f} KB)")
    return True


def update_csv_with_live_results(json_path: Path, csv_path: Path):
    if not json_path.is_file():
        return
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        return

    best_record = max(history, key=lambda x: x.get("top1", 0.0))
    m = best_record.get("metrics", {})
    snr = best_record.get("snr_breakdown", {})

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        fieldnames = reader[0].keys() if reader else []

    updated_rows = []
    for r in reader:
        if r["model_name"] == "CED-SplitFFN-MoE (Molab Run)":
            t1 = m.get('top1', 0.0)
            t1 = t1 if t1 <= 1.0 else t1 / 100.0
            r["all_top1_acc"] = f"{t1*100:.2f}%"
            t3 = m.get('top3', 0.0)
            t3 = t3 if t3 <= 1.0 else t3 / 100.0
            r["all_top3_acc"] = f"{t3*100:.2f}%"
            r["all_balanced_acc"] = f"{m.get('balanced_acc', 0.0)*100:.2f}%"
            r["all_mAP"] = f"{m.get('map', 0.0):.4f}"
            r["all_auc"] = f"{m.get('auc', 0.0):.4f}"
            r["all_macro_f1"] = f"{m.get('f1', 0.0):.4f}"
            r["mean_alpha"] = f"{m.get('mean_alpha', 0.5):.3f}"
            for snr_k in ["-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"]:
                d = snr.get(snr_k, {})
                r[f"snr_{snr_k}_acc"] = f"{d.get('top1', 0.0)*100:.2f}%"
                r[f"snr_{snr_k}_alpha"] = f"{d.get('mean_alpha', 0.5):.3f}"
            r["notes"] = f"Epoch {best_record.get('epoch')}/{len(history)} best result synchronized from Molab"
        updated_rows.append(r)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Updated {csv_path} with live Molab metrics (Best Top-1: {m.get('top1', 0.0)*100:.2f}%)")


if __name__ == "__main__":
    local_dir = Path("output/molab_sync")
    local_dir.mkdir(parents=True, exist_ok=True)

    print("--- DONG BO FILE TU MOLAB VE LOCAL ---")
    files_to_sync = [
        ("/marimo/config/config.yaml", local_dir / "config/config.yaml"),
        ("/marimo/config/models.yaml", local_dir / "config/models.yaml"),
        ("/marimo/logs/summary_50epochs.json", local_dir / "logs/summary_50epochs.json"),
    ]

    for remote_f, local_f in files_to_sync:
        download_remote_file(remote_f, local_f)

    update_csv_with_live_results(
        local_dir / "logs/summary_50epochs.json",
        Path("output/CED_SplitFFN_MoE_benchmark_results.csv"),
    )
    print("Dong bo hoan tat!")
