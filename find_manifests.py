from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0
)

code = r"""
from pathlib import Path
import pandas as pd

p = Path("/marimo/dataset")
for f in p.rglob("*.csv"):
    df = pd.read_csv(f)
    print(f.relative_to(p), f"len={len(df)}", "cols=", list(df.columns)[:5])
"""

res = c.run(code)
print(res)
