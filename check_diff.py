import difflib
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=30.0,
)

for path in ['imp/ced-tiny/ced_tiny/vit_bias.py', 'imp/ced-tiny/ced_tiny/config.py', 'imp/ced-tiny/ced_tiny/trainer.py', 'imp/ced-tiny/config/ced_tiny.yaml']:
    with open(path, 'r', encoding='utf-8') as f:
        loc = f.read()
    rem = client.run(f"with open('/marimo/{path}') as f: print(f.read())")
    diff = list(difflib.unified_diff(rem.splitlines(True), loc.splitlines(True), fromfile='remote', tofile='local'))
    if diff:
        print(f"=== DIFF: {path} ===")
        print(''.join(diff[:30]))
    else:
        print(f"=== IDENTICAL: {path} ===")
