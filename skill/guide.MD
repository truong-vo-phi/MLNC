# /output/log
- Tạo lưu trữ các hiển thị trong quá trình chạy của pipeline vào .log
- Cấu trúc lưu trữ: /output/log/<tên_experiment>-<ngày>-<tháng>-<năm>-<giờ>-<phút>-<giây>.log
Ví dụ: "experiment-01-2026-10-02-22-36-09.log"

# /output/checkpoint
- Tạo lưu trữ các checkpoint của mô hình
- Cấu trúc lưu trữ: /output/checkpoint/<tên_experiment>-<ngày>-<tháng>-<năm>-<giờ>-<phút>-<giây>/last_best_checkpoint.pt
Ví dụ: "output/checkpoint/experiment-01-2026-10-02-22-36-09/last_best_checkpoint.pt"

# /output/result
- Tạo lưu trữ các kết quả của pipeline
- Cấu trúc lưu trữ: /output/result/<tên_experiment>-<ngày>-<tháng>-<năm>-<giờ>-<phút>-<giây>/result.csv
Ví dụ: "output/result/experiment-01-2026-10-02-22-36-09/result.csv"

# về các notebook
các cell dưới đây dùng để tải dataset
```
from gdrive_fsspec import GoogleDriveFileSystem

fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
mo.output.clear_console()
```
```
# CELL: tải 2 dataset từ Kaggle (độc lập, không phụ thuộc Google Drive)
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Kaggle Access Token (kaggle.com -> Settings -> API)
# ⚠️ KHÔNG chia sẻ notebook/canvas chứa token này. Nếu gặp 401/403 → tạo lại token.
KAGGLE_ACCESS_TOKEN = "KGAT_eab1fda8096b1901c2253c3944c981c1"

# Nguồn (Kaggle)
KAGGLE_MIX_DATASET = "truongvp/36-mixture-data"  # https://www.kaggle.com/datasets/truongvp/36-mixture-data
KAGGLE_AUDIOSET = "truongvp/audioset"            # https://www.kaggle.com/datasets/truongvp/audioset
KAGGLE_VLSP = "longhoangminh/vlsp-100h"           # https://www.kaggle.com/datasets/longhoangminh/vlsp-100h

# Đích
AUDIOSET_ZIP_PATH = Path("/marimo/datasets/audioset.zip")
AUDIOSET_EXTRACT_DIR = Path("/marimo/datasets/audioset")
MIX_DATASET_ZIP_PATH = Path("/marimo/datasets/mix-dataset.zip")
MIX_DATASET_EXTRACT_DIR = Path("/marimo/datasets/mix-dataset")
VLSP_ZIP_PATH = Path("/marimo/datasets/vlsp-100h.zip")
VLSP_EXTRACT_DIR = Path("/marimo/datasets/vlsp-100h")


def download_http(
    url: str,
    destination: Path,
    description: str,
    headers: dict[str, str] | None = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> Path:
    """Stream url to destination with a byte-level progress bar."""
    if destination.exists():
        print(f"{description} already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
        resp.raise_for_status()
        total_bytes = int(resp.headers.get("content-length", 0)) or None

        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
        ) as progress, destination.open("wb") as dst:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                dst.write(chunk)
                progress.update(len(chunk))

    size_gb = destination.stat().st_size / (1024**3)
    print(f"{description} downloaded: {destination} ({size_gb:.2f} GB)")
    return destination


def download_and_extract_kaggle_dataset(
    dataset_slug: str,
    zip_path: Path,
    extract_dir: Path,
    description: str,
) -> Path:
    """Download a Kaggle dataset ZIP and extract it (skips work already done)."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"{description} already extracted: {extract_dir}")
        return extract_dir

    if not zip_path.exists():
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
        headers = {"Authorization": f"Bearer {KAGGLE_ACCESS_TOKEN}"}
        download_http(url, zip_path, description=f"{description}.zip", headers=headers)

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid ZIP file: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in tqdm(
            zip_ref.infolist(),
            desc=f"Extracting {description}",
            unit="file",
        ):
            zip_ref.extract(member, extract_dir)

    print(f"{description} extracted: {extract_dir}")
    return extract_dir


def download_and_extract_audioset() -> Path:
    """Download and extract the AudioSet dataset from Kaggle."""
    return download_and_extract_kaggle_dataset(
        KAGGLE_AUDIOSET, AUDIOSET_ZIP_PATH, AUDIOSET_EXTRACT_DIR,
        description="AudioSet",
    )


def download_and_extract_mix_dataset() -> Path:
    """Download and extract the mix-dataset from Kaggle."""
    return download_and_extract_kaggle_dataset(
        KAGGLE_MIX_DATASET, MIX_DATASET_ZIP_PATH, MIX_DATASET_EXTRACT_DIR,
        description="mix-dataset",
    )


def download_and_extract_vlsp() -> Path:
    """Download and extract the VLSP-100h dataset from Kaggle."""
    return download_and_extract_kaggle_dataset(
        KAGGLE_VLSP, VLSP_ZIP_PATH, VLSP_EXTRACT_DIR,
        description="VLSP-100h",
    )


audioset_dir = download_and_extract_audioset()
mix_dataset_dir = download_and_extract_mix_dataset()
# vlsp_dir = download_and_extract_vlsp()
```