import os, hashlib

files_to_check = [
    'imp/ced-tiny/ced_tiny/cli.py',
    'imp/ced-tiny/ced_tiny/config.py',
    'imp/ced-tiny/ced_tiny/data.py',
    'imp/ced-tiny/ced_tiny/evaluate.py',
    'imp/ced-tiny/ced_tiny/frontend.py',
    'imp/ced-tiny/ced_tiny/loss.py',
    'imp/ced-tiny/ced_tiny/model.py',
    'imp/ced-tiny/ced_tiny/pipeline.py',
    'imp/ced-tiny/ced_tiny/runtime.py',
    'imp/ced-tiny/ced_tiny/trainer.py',
    'imp/ced-tiny/ced_tiny/vit_bias.py',
    'imp/ced-tiny/config/ced_tiny.yaml'
]

hashes = {}
for rel in files_to_check:
    full = os.path.join('/marimo', rel)
    if os.path.exists(full):
        with open(full, 'rb') as f:
            hashes[rel] = hashlib.md5(f.read()).hexdigest()
    else:
        hashes[rel] = None
import json
print(json.dumps(hashes))
