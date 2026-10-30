from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0
)

code = r"""
from pathlib import Path
import numpy as np

vd = Path("/marimo/dataset/cache/validation.done")
vn = Path("/marimo/dataset/cache/validation.npy")
td = Path("/marimo/dataset/cache/train.done")
tn = Path("/marimo/dataset/cache/train.npy")

print("td size:", td.stat().st_size, "lines:", len(td.read_text().splitlines()))
print("vd size:", vd.stat().st_size, "lines:", len(vd.read_text().splitlines()))

v_data = np.load(vn, mmap_mode="r")
print("vn shape:", v_data.shape)

t_data = np.load(tn, mmap_mode="r")
print("tn shape:", t_data.shape)

# Check manifests in dataset
mix_dir = Path("/marimo/dataset/mix-dataset/mix-dataset-v2")
print("mix_dir exists:", mix_dir.exists())
if mix_dir.exists():
    for f in mix_dir.glob("*.csv"):
        print(f.name, len(f.read_text().splitlines()) - 1)
"""

res = c.run(code)
print(res)
