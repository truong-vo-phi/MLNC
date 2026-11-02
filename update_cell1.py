import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

code_cell1 = '''
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download
import shutil

# 1. Cấu hình
REPO_ID = "phitruong0606/mix-dataset-v2"
FILENAME = "mix-dataset-v2.zip"

DOWNLOAD_DIR = Path(r"zip/")
EXTRACT_DIR = Path(r"datasets/")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

zip_path = None
bad_file = None

# Kiểm tra nếu dataset đã có sẵn đầy đủ
_eval_manifest = EXTRACT_DIR / "mix-dataset-v2" / "eval" / "manifest.csv"
if _eval_manifest.is_file():
    print(f"✓ Dataset đã tồn tại tại {EXTRACT_DIR}. Sẵn sàng sử dụng!")
else:
    # 2. Tải file ZIP
    print("Đang tải dataset...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=FILENAME,
        local_dir=str(DOWNLOAD_DIR),
    )
    print(f"Đã tải: {zip_path}")

    # 3. Kiểm tra và giải nén
    print("Đang kiểm tra file ZIP...")
    with ZipFile(zip_path, "r") as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file is not None:
            raise RuntimeError(f"File ZIP bị lỗi tại: {bad_file}")
        print("File ZIP hợp lệ.")
        print("Đang giải nén...")
        zip_ref.extractall(EXTRACT_DIR)
    print(f"Hoàn tất! Dataset nằm tại: {EXTRACT_DIR}")
'''

update_script = f"""
import marimo._code_mode as cm

async def update_cell1():
    async with cm.get_context() as ctx:
        ctx.edit_cell("dSpg", {repr(code_cell1.strip())})
        ctx.run_cell("dSpg")

await update_cell1()
"""

rc = execute_remote(update_script)
print(f"Cell 1 update executed with code {rc}")
