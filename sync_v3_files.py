import base64
from pathlib import Path
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=60.0,
)

files_to_sync = [
    'imp/ced-tiny/ced_tiny/vit_bias.py',
    'imp/ced-tiny/ced_tiny/config.py',
    'imp/ced-tiny/ced_tiny/trainer.py',
    'imp/ced-tiny/config/ced_tiny.yaml'
]

for rel in files_to_sync:
    p = Path(rel)
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode('ascii')
    code = f"""
import base64
from pathlib import Path
target = Path('/marimo/{rel}')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(base64.b64decode('{b64}'))
print(f'Synced /marimo/{rel} ({{target.stat().st_size}} bytes)')
"""
    out = client.run(code)
    print(out.strip())
