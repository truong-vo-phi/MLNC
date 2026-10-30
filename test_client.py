import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.molab_deploy import MolabClient

URL = "https://sb-d3f93608a9dca958.sb.molab.run"
TOKEN = "cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9"

client = MolabClient(url=URL, token=TOKEN, timeout=60.0)
out = client.run("import sys; print(f'Sandbox Python: {sys.version}')")
print("Output from sandbox:")
print(out)
