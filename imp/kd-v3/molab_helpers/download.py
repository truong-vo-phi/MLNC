import os, sys, time
from pathlib import Path
import requests
from tqdm import tqdm
sys.path.insert(0, "/marimo")
from scripts import download_dataset as dl

root = Path("/marimo/datasets")
extract_dir = root / "mix-dataset"
zip_path = root / "mix-dataset.zip"
part = root / "mix-dataset.zip.part"
url = f"{dl.KAGGLE_API}/{dl.resolve_slug('kaggle-mix-dataset')}"
headers = {"Authorization": f"Bearer {dl.read_token()}"}
root.mkdir(parents=True, exist_ok=True)
Path("/marimo/output").mkdir(parents=True, exist_ok=True)

if not (extract_dir.exists() and any(extract_dir.iterdir())) and not zip_path.exists():
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        h = dict(headers)
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, stream=True, headers=h, timeout=60) as r:
                r.raise_for_status()
                resumed = bool(have) and r.status_code == 206
                if have and not resumed:
                    print("server khong ho tro Range, tai lai tu dau", flush=True)
                total = int(r.headers.get("content-length", 0)) + (have if resumed else 0)
                print(f"[tai] lan {attempt + 1}: bat dau tu {have if resumed else 0} / tong {total} byte", flush=True)
                with part.open("ab" if resumed else "wb") as f, tqdm(total=total or None, initial=have if resumed else 0, unit="B", unit_scale=True, unit_divisor=1024, desc="mix-dataset.zip", mininterval=20) as bar:
                    for chunk in r.iter_content(chunk_size=dl.CHUNK_SIZE):
                        f.write(chunk)
                        bar.update(len(chunk))
            part.rename(zip_path)
            break
        except Exception as error:
            print("[tai] loi, thu lai sau 10s:", repr(error), flush=True)
            time.sleep(10)
    else:
        raise SystemExit("tai that bai sau 30 lan thu")

if not (extract_dir.exists() and any(extract_dir.iterdir())):
    print("[giai nen]", zip_path, flush=True)
    dl.extract_zip(zip_path, extract_dir)
    zip_path.unlink()
data_root = dl.find_root(extract_dir, "labels.txt")
Path("/marimo/output/download.done").write_text(str(data_root))
print("[xong]", data_root, flush=True)
