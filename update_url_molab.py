from pathlib import Path
from scripts.molab_deploy import MolabClient

client = MolabClient(
    url="https://sb-d3f93608a9dca958.sb.molab.run",
    token="cc93d32e13723437e102b78db105acc09bc41deae062f21ed920cb0e9035e4d9",
    timeout=30.0,
)

code = """
import yaml
from pathlib import Path
p = Path('/marimo/config/url.yaml')
data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
data['mixture-36l'] = '/marimo/dataset/mix-dataset/mix-dataset-v2'
p.write_text(yaml.safe_dump(data), encoding='utf-8')
print('Updated /marimo/config/url.yaml:', p.read_text())
"""
print(client.run(code))
