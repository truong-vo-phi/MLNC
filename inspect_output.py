import sys
from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0
)

code = r"""
from pathlib import Path
p = Path("/marimo/output/kd-v5")
if p.exists():
    for f in p.rglob("*"):
        if f.is_file():
            print(f, f"({f.stat().st_size} bytes)")
else:
    print("Not found")
"""

res = c.run(code)
print(res)
