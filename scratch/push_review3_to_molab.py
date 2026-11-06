import base64
from pathlib import Path
import subprocess

zip_bytes = Path("scratch/review3_deploy.zip").read_bytes()
b64_str = base64.b64encode(zip_bytes).decode("ascii")

remote_code = f"""import base64, io, zipfile, shutil
from pathlib import Path

target_dir = Path('/marimo/review-3')
if target_dir.exists():
    shutil.rmtree(target_dir)
target_dir.mkdir(parents=True, exist_ok=True)

zip_data = base64.b64decode('{b64_str}')
with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
    z.extractall(target_dir)

print('Extracted review-3 files to:', target_dir)
print('Installed files count:', len(list(target_dir.rglob('*'))))
"""

cmd = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"d:/PROJECT/MLNC/.agents/skills/marimo-pair/scripts/execute-code.sh",
    "--url", "https://sb-3632b8f8da56ce54.sb.molab.run/",
    "--token", "1c3cd423d853201370f94f61624c8bd1cca4002a61f133c8178c7fbcfb4509b1",
    "-",
]

result = subprocess.run(cmd, input=remote_code, capture_output=True, text=True)
print("Return code:", result.returncode)
print("Stdout:", result.stdout)
if result.stderr:
    print("Stderr:", result.stderr[-300:])
