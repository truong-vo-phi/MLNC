import subprocess
from pathlib import Path

b64 = Path("scratch/dual_ced_b64.txt").read_text().strip()
remote_code = f"""import base64, io, zipfile
from pathlib import Path
b64 = '''{b64}'''
buf = io.BytesIO(base64.b64decode(b64))
dest = Path('/marimo/dual_ced_tiny')
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(buf) as z:
    z.extractall(dest)
print('SUCCESS: Extracted to /marimo/dual_ced_tiny:', [p.name for p in sorted(dest.iterdir())])
"""
Path("scratch/remote_code.py").write_text(remote_code, encoding="utf-8")
print("Wrote scratch/remote_code.py, size:", len(remote_code))

cmd = [
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "D:/PROJECT/MLNC/.agents/skills/marimo-pair/scripts/execute-code.sh",
    "--url", "https://sb-2191dc50056e2d63.sb.molab.run/",
    "--token", "f99a48dc892d7f816c32bd3cd6662e7d30b2e40245e16cddc661bed80e0ef147",
    "scratch/remote_code.py"
]
p = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", p.stdout)
print("STDERR:", p.stderr)
