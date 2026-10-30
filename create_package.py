import os
import zipfile
from pathlib import Path

base_dir = Path(r"D:\PROJECT\MLNC")
out_zip = base_dir / "package_to_molab.zip"

include_dirs = ["src", "scripts", "config", "imp/ced-tiny", "imp/ft-beat", "imp/ema-beat"]
include_files = ["partition_labels_36.json", "rules.MD", "pyproject.toml"]

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_dir in include_dirs:
        abs_dir = base_dir / rel_dir
        for root, dirs, files in os.walk(abs_dir):
            if "__pycache__" in root or ".pytest_cache" in root or ".ruff_cache" in root:
                continue
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full_path = Path(root) / f
                rel_path = full_path.relative_to(base_dir)
                zf.write(full_path, str(rel_path))
    for f in include_files:
        full_path = base_dir / f
        if full_path.is_file():
            zf.write(full_path, f)

print(f"Created {out_zip} with size {out_zip.stat().st_size} bytes")
