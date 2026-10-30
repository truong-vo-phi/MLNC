import sys
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=600.0,
)

if len(sys.argv) > 1:
    filename = sys.argv[1]
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
else:
    code = sys.stdin.read()

res = client.run(code)
print(res)
