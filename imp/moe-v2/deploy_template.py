import base64
import io
import os
import shutil
import zipfile

b64_data = """{B64_DATA}"""

dest_dir = "/marimo/moe-v2"
csv_backup = None
csv_path = os.path.join(dest_dir, "result.csv")
if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 50:
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_backup = f.read()

os.makedirs(dest_dir, exist_ok=True)

bio = io.BytesIO(base64.b64decode(b64_data))
with zipfile.ZipFile(bio, "r") as z:
    z.extractall(dest_dir)

if csv_backup is not None:
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_backup)
    print("Preserved existing result.csv with", len(csv_backup.splitlines()), "rows.")

print("Successfully deployed moe-v2 to sandbox at:", dest_dir)
print("Files in /marimo/moe-v2:")
for root, dirs, files in os.walk(dest_dir):
    print(" ", root, files)
