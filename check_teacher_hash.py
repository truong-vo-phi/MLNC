from scripts.molab_deploy import MolabClient

c = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0
)

code = r"""
import hashlib
from pathlib import Path
p = Path("/marimo/pretrained/beats_teacher_fold0_weights.pt")
h = hashlib.sha256(p.read_bytes()).hexdigest()
print("Teacher sha256:", h)
"""

res = c.run(code)
print(res)
