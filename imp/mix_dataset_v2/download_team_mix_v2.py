"""Download and verify the team's pinned Hugging Face dataset/cache assets."""
from pathlib import Path
import hashlib
import json
import json
import os
import shutil
import time
import zipfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "benchmark_results/team_kd_moe_research_20261102"
DATA = ROOT / "datasets/team_mix_dataset_v2"
FILES = ("labels.txt", "beats_teacher_ema_standalone.pt", "beats_teacher_fold0_weights.pt",
         "best_e037_20261020T203958Z.pt", "mix-dataset-v2.zip", "cache-nhat.zip")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def download_ranges(repo, revision, name, folder, size):
    """Resume bounded HTTP ranges when the full-object CDN stream stalls."""
    target = folder / name
    if target.is_file() and target.stat().st_size == size:
        return target
    partial = folder / (name + ".range-partial")
    progress = folder / (name + ".range-progress.json")
    chunk_size = 16 * 1024 * 1024
    count = (size + chunk_size - 1) // chunk_size
    completed = set()
    if progress.exists() and partial.exists():
        saved = json.loads(progress.read_text())
        if saved["revision"] == revision and saved["size"] == size and saved["chunk_size"] == chunk_size:
            completed.update(saved["completed"])
    if not partial.exists():
        with partial.open("wb") as stream:
            stream.truncate(size)
    lock = threading.Lock()
    started = time.monotonic()
    def fetch(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        for attempt in range(6):
            try:
                url = (f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
                       f"?download=true&part={start}&retry={attempt}")
                request = urllib.request.Request(url, headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Unexpected range response for {name} part {index}")
                    body = response.read(end-start+2)
                if len(body) != end-start+1:
                    raise ValueError("Incomplete ranged download")
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(body)
                with lock:
                    completed.add(index)
                    progress.write_text(json.dumps(dict(revision=revision, size=size, chunk_size=chunk_size,
                                                        completed=sorted(completed))))
                    if len(completed) % 20 == 0 or len(completed) == count:
                        print(f"RANGE {name} {len(completed)}/{count} elapsed={time.monotonic()-started:.0f}s", flush=True)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(30, 2**attempt))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for _ in pool.map(fetch, [i for i in range(count) if i not in completed]):
            pass
    partial.replace(target)
    return target


def main():
    RESEARCH.mkdir(parents=True, exist_ok=True)
    revision = json.loads((ROOT / "config/hf_dataset_revision.json").read_text())["revision"]
    tree = {entry["path"]:entry for entry in json.loads((ROOT / "config/hf_dataset_files.json").read_text())}
    downloads = DATA / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    status_file = RESEARCH / "data_download_status.json"
    status = dict(repo="phitruong0606/mix-dataset-v2", revision=revision,
                  root=str(DATA), status="running", downloaded=[], extracted=[])
    def save():
        status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        for name in FILES:
            status["active_file"] = name
            save()
            print(f"DOWNLOAD {name} expected_GiB={tree[name]['size']/2**30:.3f}", flush=True)
            if name.endswith(".zip"):
                path = download_ranges(status["repo"], revision, name, downloads, tree[name]["size"])
            else:
                path = Path(hf_hub_download(status["repo"], filename=name, repo_type="dataset",
                                           revision=revision, local_dir=downloads))
            if path.stat().st_size != tree[name]["size"]:
                raise ValueError(f"Size mismatch: {name}")
            checksum = sha256(path)
            expected = tree[name].get("lfs", {}).get("oid")
            if expected and checksum != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
            status["downloaded"].append(dict(file=name, path=str(path), bytes=path.stat().st_size,
                                             sha256=checksum, lfs_sha256_verified=bool(expected)))
            save()
            print(f"VERIFIED {name} SHA256={checksum}", flush=True)
        extracted = DATA / "extracted"
        extracted.mkdir(exist_ok=True)
        for name in ("mix-dataset-v2.zip", "cache-nhat.zip"):
            status["active_file"] = f"extract:{name}"
            save()
            with zipfile.ZipFile(downloads / name) as archive:
                entries = archive.infolist()
                total = sum(entry.file_size for entry in entries)
                free = shutil.disk_usage(extracted).free
                print(f"ARCHIVE {name} files={len(entries)} expanded_GiB={total/2**30:.3f} free_GiB={free/2**30:.3f}", flush=True)
                listing = [dict(path=entry.filename, size=entry.file_size, compressed=entry.compress_size)
                           for entry in entries]
                (RESEARCH / (name + ".contents.json")).write_text(json.dumps(listing, indent=2), encoding="utf-8")
                if free < total + 3*2**30:
                    raise RuntimeError(f"Insufficient free space to extract {name}; archive is downloaded and verified")
                destination = extracted.resolve()
                for entry in entries:
                    target = (destination / entry.filename).resolve()
                    if not target.is_relative_to(destination):
                        raise ValueError("Unsafe archive path")
                start = time.monotonic()
                for number, entry in enumerate(entries, 1):
                    # ZipFile checks CRC while reading each extracted entry.
                    archive.extract(entry, destination)
                    if number % 10000 == 0:
                        print(f"EXTRACT {name} {number}/{len(entries)} elapsed={time.monotonic()-start:.0f}s", flush=True)
                status["extracted"].append(dict(archive=name, root=str(destination),
                                                entries=len(entries), uncompressed_bytes=total, crc_checked=True))
                save()
        status["status"] = "complete"
        status["active_file"] = None
        save()
        print("DOWNLOAD_AND_EXTRACTION_COMPLETE", flush=True)
    except Exception as error:
        status["status"] = "error"
        status["error"] = dict(type=type(error).__name__, message=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
