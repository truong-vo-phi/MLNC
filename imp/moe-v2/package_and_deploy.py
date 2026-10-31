import base64
import io
import os
import zipfile
from pathlib import Path

CURRENT_DIR = Path("D:/PROJECT/MLNC/imp/moe-v2")

def build_package():
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(CURRENT_DIR):
            rel_root = os.path.relpath(root, CURRENT_DIR)
            if any(p in rel_root for p in [".git", "__pycache__", "checkpoints", ".pytest_cache"]):
                continue
            for f in files:
                if f.endswith((".b64", ".pt", ".zip", "package_and_deploy.py")):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, CURRENT_DIR)
                z.write(full_path, rel_path)
    
    b64_str = base64.b64encode(bio.getvalue()).decode("utf-8")
    with open(CURRENT_DIR / "package.b64", "w", encoding="utf-8") as f:
        f.write(b64_str)
    
    with open(CURRENT_DIR / "deploy_template.py", "r", encoding="utf-8") as f:
        template = f.read()
    
    deploy_code = template.replace("{B64_DATA}", b64_str)
    with open(CURRENT_DIR / "deploy_to_sandbox.py", "w", encoding="utf-8") as f:
        f.write(deploy_code)
    
    print(f"Packaged {len(b64_str)} chars to package.b64 and deploy_to_sandbox.py")

if __name__ == "__main__":
    build_package()
