from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0
)

code = r"""
from pathlib import Path

mix_dir = Path("/marimo/dataset/mix-dataset/mix-dataset-v2")
for f in mix_dir.rglob("*"):
    if f.is_file():
        print(f.relative_to(mix_dir), f.stat().st_size)
"""

res = c.run(code)
print(res)
