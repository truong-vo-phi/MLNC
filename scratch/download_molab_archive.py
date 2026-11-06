import subprocess
import base64
import zipfile
import shutil
from pathlib import Path

DEST_DIR = Path("d:/PROJECT/MLNC/molab_clone")
DEST_DIR.mkdir(parents=True, exist_ok=True)
ZIP_LOCAL = DEST_DIR / "molab_code_and_text.zip"

CHUNK_SIZE = 512 * 1024  # 512 KB binary chunk (< 1MB Marimo buffer limit)

# Get total file size on sandbox
get_size_code = 'import os; print(os.stat("/tmp/molab_code_and_text.zip").st_size)'
total_size = int(subprocess.check_output(["python", "scratch/py_exec.py", "-c", get_size_code], text=True).strip())
print(f"Total size to download: {total_size / (1024*1024):.2f} MB")

with open(ZIP_LOCAL, "wb") as out_f:
    offset = 0
    while offset < total_size:
        length = min(CHUNK_SIZE, total_size - offset)
        chunk_code = f'''import base64
with open("/tmp/molab_code_and_text.zip", "rb") as f:
    f.seek({offset})
    data = f.read({length})
print(base64.b64encode(data).decode())
'''
        res = subprocess.check_output(["python", "scratch/py_exec.py", "-c", chunk_code], text=True)
        # Strip all whitespaces/newlines from base64 string
        clean_b64 = "".join(res.split())
        data = base64.b64decode(clean_b64)
        out_f.write(data)
        offset += len(data)
        print(f"Downloaded {offset / (1024*1024):.2f} / {total_size / (1024*1024):.2f} MB ({(offset/total_size)*100:.1f}%)")

print("Download complete! Unzipping...")
with zipfile.ZipFile(ZIP_LOCAL, 'r') as zip_ref:
    zip_ref.extractall(DEST_DIR)

print(f"Successfully extracted all files to {DEST_DIR}!")
